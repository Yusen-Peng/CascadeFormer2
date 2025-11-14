import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from base_dataset import ActionRecognitionDataset
from data_preprocessing import PoseSegDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_train_val(train_seq, train_lbl, val_ratio=0.15, seed=42):
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_seq)),
        test_size=val_ratio,
        random_state=seed,
        stratify=train_lbl
    )
    tr_seq  = [train_seq[i] for i in tr_idx]
    tr_lbl  = [train_lbl[i] for i in tr_idx]
    val_seq = [train_seq[i] for i in val_idx]
    val_lbl = [train_lbl[i] for i in val_idx]

    return tr_seq, tr_lbl, val_seq, val_lbl


def collate_fn_batch_padding(batch):
    """
    a collate function for DataLoader that pads sequences to the maximum length in the batch.
    
    Returns:
      padded_seqs: (B, T_max, D) tensor
      labels: (B,) or (B, something)
      lengths: list of original sequence lengths
    """
    sequences, labels = zip(*batch)    
    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    return padded_seq, labels

def pose_seg_to_sequences(
    pose_ds: PoseSegDataset,
    abnormal_label: int = 1,
    normal_label: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Convert PoseSegDataset (N, C, T, V) into Penn-style:
        sequences: list of (T, D) np arrays
        labels:    list[int] with 0/1 labels

    - Uses pose_ds.get_all_data(normalize_pose_segs=True) so we keep the same normalization
    - Flattens (C, T, V) -> (T, C*V)
    """
    # Get normalized data in shape (N, C, T, V) or list of (C, T, V)
    data_list = pose_ds.get_all_data(normalize_pose_segs=True)

    # get_all_data returns a list if num_transform == 1 or eval-mode
    if isinstance(data_list, np.ndarray):
        # shape: (N, C, T, V)
        data_list = [data_list[i] for i in range(data_list.shape[0])]

    sequences: List[np.ndarray] = []
    labels: List[int] = []

    raw_labels = pose_ds.labels.astype(int)  # +1 (normal) or -1 (abnormal)

    for arr, lab in zip(data_list, raw_labels):
        # arr: (C, T, V)
        C, T, V = arr.shape
        # → (T, V, C) → (T, V*C)
        seq = arr.transpose(1, 2, 0).reshape(T, -1).astype(np.float32)
        sequences.append(seq)

        if lab == -1:
            labels.append(abnormal_label)
        else:
            labels.append(normal_label)

    return sequences, labels


if __name__ == "__main__":
    set_seed(42)

    # ---- 1) Build UBnormal PoseSegDataset for train/test ----
    # You already have args somewhere for UBnormal; adjust as needed.
    # Example skeleton:
    dataset_args = {
        'headless': False,
        'scale': False,
        'scale_proportional': True,
        'seg_len': 24,
        'return_indices': True,
        'return_metadata': True,
        'dataset': "UBnormal",
        'train_seg_conf_th': 0.0,
        'specific_clip': None,
        'seg_stride': 1,
    }

    pose_path = {
        "train":      "UBnormal/pose/train",
        "test":       "UBnormal/pose/test",
    }
    vid_path = {
        "train": "UBnormal/gt",
        "test":  "UBnormal/gt",
    }
    pose_path_train_abnormal = "UBnormal/pose/abnormal_train"

    # Train set (with abnormal_train_path)
    train_pose_ds = PoseSegDataset(
        path_to_json_dir=pose_path["train"],
        path_to_vid_dir=vid_path["train"],
        normalize_pose_segs=True,
        evaluate=False,
        abnormal_train_path=pose_path_train_abnormal,
        **dataset_args,
    )

    # Test set
    test_pose_ds = PoseSegDataset(
        path_to_json_dir=pose_path["test"],
        path_to_vid_dir=vid_path["test"],
        normalize_pose_segs=True,
        evaluate=True,
        abnormal_train_path=None,
        **dataset_args,
    )

    # ---- 2) Convert to Penn-style sequences + labels ----
    train_seq, train_lbl = pose_seg_to_sequences(train_pose_ds)
    test_seq,  test_lbl  = pose_seg_to_sequences(test_pose_ds)

    # Optional train/val split just like Penn
    train_seq, train_lbl, val_seq, val_lbl = split_train_val(
        train_seq, train_lbl, val_ratio=0.15, seed=42
    )

    # ---- 3) Wrap with ActionRecognitionDataset ----
    train_ds = ActionRecognitionDataset(train_seq, train_lbl)
    val_ds   = ActionRecognitionDataset(val_seq,   val_lbl)
    test_ds  = ActionRecognitionDataset(test_seq,  test_lbl)

    # ---- 4) DataLoaders using the SAME collate_fn as Penn ----
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn_batch_padding,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn_batch_padding,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn_batch_padding,
    )

    # ---- 5) Sanity check ----
    print("UBnormal train size:", len(train_loader.dataset))
    x_batch, y_batch = next(iter(train_loader))
    print("Example UBnormal batch shape:", x_batch.shape)   # (B, T_max, D)
    print("Example UBnormal labels shape:", y_batch.shape)
