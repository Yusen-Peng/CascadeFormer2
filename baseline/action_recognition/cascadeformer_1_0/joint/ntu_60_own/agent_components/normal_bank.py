import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from typing import Dict, List, Optional
from .constants import classify_labels, normal_action_labels, model_config
from .perceiver import CascadeFormerWrapper

def build_normal_bank(
    model: CascadeFormerWrapper,
    dataset,
    batch_size: int = 32,
    device: str = "cuda",
    per_class: bool = False,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Build a bank of embeddings using ONLY normal actions defined in `normal_action_labels`.

    Returns:
        - If per_class == False:
            np.ndarray of shape (N_normals, D)
        - If per_class == True:
            dict[int, np.ndarray], where keys are NORMAL class IDs
    """
    # Map the normal action names to their class indices once.
    normal_name_to_id = {name: i for i, name in enumerate(classify_labels)}
    normal_class_ids = {normal_name_to_id[name] for name in normal_action_labels}
    # (Optional) sanity: ensure disjoint from abnormal list if needed
    # assert not any(id_ in normal_class_ids for id_ in abnormal_class_ids)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if per_class:
        # Only allocate bins for NORMAL classes
        banks: Dict[int, List[np.ndarray]] = {c: [] for c in sorted(normal_class_ids)}
    else:
        embeddings: List[np.ndarray] = []

    with torch.inference_mode():
        for skeletons, labels, _ in tqdm(loader):
            skeletons = skeletons.to(device)

            # Preprocessing sequences from CTR-GCN-style input
            B, C, T, V, M = skeletons.shape
            sequences = skeletons.permute(0, 2, 3, 1, 4)

            # Select most active person (M=1)
            motion = sequences.abs().sum(dim=(1, 2, 3))  # (B, M)
            main_person_idx = motion.argmax(dim=-1)       # (B,)

            indices = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
            sequences = torch.gather(sequences, dim=4, index=indices).squeeze(-1)  # (B, T, V, C)
            skeletons = sequences.float().to(device)  # (B, T, J, D)

            # Forward pass
            out = model.infer(skeletons)         # dict with "embedding" (B, D) as np.ndarray
            emb_batch = out["embedding"]          # (B, D), numpy array

            # Figure out which items in this batch are NORMAL
            labels_np = labels.detach().cpu().numpy()
            keep_idx = [i for i, lbl in enumerate(labels_np) if int(lbl) in normal_class_ids]
            if not keep_idx:
                continue

            emb_norm = emb_batch[keep_idx]        # (B_norm, D)
            lbl_norm = labels_np[keep_idx]        # (B_norm,)

            if per_class:
                for e, lbl in zip(emb_norm, lbl_norm):
                    banks[int(lbl)].append(e)
            else:
                embeddings.append(emb_norm)

    if per_class:
        # Stack each normal class bank; drop classes with zero samples
        stacked = {
            c: np.stack(v, axis=0).astype(np.float32)
            for c, v in banks.items() if len(v) > 0
        }

        return stacked
    else:
        if len(embeddings) == 0:
            # ; return an empty (0, D) array
            print("❌error: No normal samples found")
            return np.zeros((0, model_config["hidden_size"]), dtype=np.float32)
        return np.concatenate(embeddings, axis=0).astype(np.float32)