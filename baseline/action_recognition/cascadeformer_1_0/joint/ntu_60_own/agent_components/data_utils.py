import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Any, Dict
from NTU_feeder import Feeder
from .constants import DATA_PATH, WINDOW_SIZE

def prepare_one_sample(json_path: str, shuffle: bool, train: bool):
    split = 'train' if train else 'test'
    dataset = Feeder(
        data_path=DATA_PATH,
        split=split,
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=WINDOW_SIZE,
        normalization=False,
        random_rot=False,
        p_interval=[0.5, 1],
        vel=False,
        bone=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    skeletons, labels, _ = next(iter(loader))  # (1,C,T,V,M)
    B, C, T, V, M = skeletons.shape

    sequences = skeletons.permute(0, 2, 3, 1, 4)
    # pick most active person
    motion = sequences.abs().sum(dim=(1, 2, 3))
    main_person_idx = motion.argmax(dim=-1)
    idx = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
    sequences = torch.gather(sequences, dim=4, index=idx).squeeze(-1)  # (1,T,V,C)
    window = sequences[0].cpu().numpy().astype(float)                  # (T,V,C)

    with open(json_path, "w") as f:
        json.dump(window.tolist(), f)
    return json_path, int(labels), window


def select_main_person_batch(skeletons: torch.Tensor) -> torch.Tensor:
    """
    (B,C,T,V,M) -> (B,T,V,C) using your 'most active person' heuristic.
    """
    B, C, T, V, M = skeletons.shape
    sequences = skeletons.permute(0, 2, 3, 1, 4)              # (B,T,V,C,M)
    motion = sequences.abs().sum(dim=(1, 2, 3))               # (B,M)
    main_person_idx = motion.argmax(dim=-1)                   # (B,)
    idx = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
    sequences = torch.gather(sequences, dim=4, index=idx).squeeze(-1)  # (B,T,V,C)
    return sequences


def extract_state_features(event: Dict[str,Any], scores: Dict[str,Any]) -> np.ndarray:
    """
    Aligns with training features: [entropy, knn_dist, mahalanobis, top1_conf]
    """
    ent = float(event["entropy"])
    knn = float(scores["knn_dist"])
    maha = float(scores["mahalanobis"])
    top1 = float(scores["top1_conf"])
    return np.array([ent, knn, maha, top1], dtype=float)