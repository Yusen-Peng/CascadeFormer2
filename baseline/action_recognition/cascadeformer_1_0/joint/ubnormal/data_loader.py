import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
import glob
import json
from typing import List, Tuple, Optional
from base_dataset import ActionRecognitionDataset

NUM_JOINTS_UBNORMAL = 17  # COCO-style keypoints
TARGET_LEN = 128

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def keypoints_to_flat_xy(kps: list) -> np.ndarray:
    """
    UBnormal AlphaPose format:
      kps: flat list of length 51 = 17 * (x, y, score)
    Return: (34,) = 17 * (x, y)
    """
    arr = np.array(kps, dtype=np.float32).reshape(-1, 3)  # (17,3)
    xy = arr[:, :2]                                       # (17,2)
    return xy.reshape(-1)                                 # (34,)


def load_json_pose(
    json_path: str,
    min_len: int = 1,
    target_len: int = TARGET_LEN,   # fixed number of frames after main-char selection
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load a UBnormal AlphaPose JSON and turn it into a (T, 34) sequence.

    Strategy:
      1) Among all tracks, pick the one with the longest length (main character).
      2) AFTER that, uniformly resample this track to `target_len` frames.
      3) Label from filename prefix: normal* -> 0, abnormal* -> 1.
    """
    fname = os.path.basename(json_path)
    label = 1 if fname.startswith("abnormal") else 0

    with open(json_path, "r") as f:
        data = json.load(f)   # track_id(str) -> frame_str -> {keypoints, scores}

    best_seq = None     # (T, 34) before resampling
    best_T = 0          # original length

    # --------- FIRST: choose main character (longest track) ---------
    for _, frames in data.items():
        # keep original keys, sorted numerically
        frame_keys = sorted(frames.keys(), key=lambda x: int(x))
        T = len(frame_keys)
        if T < min_len:
            continue

        poses = np.zeros((T, NUM_JOINTS_UBNORMAL * 2), dtype=np.float32)  # (T,34)

        for i, fr_key in enumerate(frame_keys):
            kps = frames[fr_key]["keypoints"]
            poses[i] = keypoints_to_flat_xy(kps)

        if T > best_T:
            best_T = T
            best_seq = poses

    # --------- THEN: resample the chosen track ---------
    if best_seq is None:
        return None, None

    if target_len is not None and best_T > 0:
        # uniformly sample indices in [0, best_T-1]
        idx = np.linspace(0, best_T - 1, num=target_len).astype(int)
        best_seq = best_seq[idx]   # (target_len, 34)

    return best_seq.astype(np.float32), int(label)


def _collect_split(root_pose_dir: str, split: str
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Collect sequences + labels from a given split under UBnormal/pose/*

    split in {"train", "abnormal_train", "test"}
    """
    pose_dir = os.path.join(root_pose_dir, split)
    json_files = sorted(
        glob.glob(os.path.join(pose_dir, "*_alphapose_tracked_person.json"))
    )

    seqs: List[np.ndarray] = []
    lbls: List[int] = []

    for jp in json_files:
        seq, lbl = load_json_pose(jp)
        if seq is None:
            continue
        seqs.append(seq)
        lbls.append(lbl)

    print(f"[{split}] clips={len(seqs)} "
          f"| abnormal={sum(lbls)} | normal={len(lbls)-sum(lbls)}")
    return seqs, lbls


def build_ubnormal_lists(root: str
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Follow Penn Action style:

    Returns:
        train_seq, train_lbl, test_seq, test_lbl

    where each seq is (T, 34) and label in {0 (normal), 1 (abnormal)}.
    """
    pose_root = os.path.join(root, "pose")

    # training data: normal train + abnormal_train
    train_seq_norm, train_lbl_norm = _collect_split(pose_root, "train")
    train_seq_ab,   train_lbl_ab   = _collect_split(pose_root, "abnormal_train")

    train_seq = train_seq_norm + train_seq_ab
    train_lbl = train_lbl_norm + train_lbl_ab

    # shuffle training data
    combined = list(zip(train_seq, train_lbl))
    random.shuffle(combined)
    train_seq[:], train_lbl[:] = zip(*combined)

    # test data
    test_seq, test_lbl = _collect_split(pose_root, "test")

    print(f"#train clips={len(train_seq)} | #test clips={len(test_seq)}")
    return train_seq, train_lbl, test_seq, test_lbl


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



if __name__ == "__main__":
    set_seed(42)
    root = "UBnormal"

    train_seq, train_lbl, test_seq, test_lbl = build_ubnormal_lists(root)
    train_seq, train_lbl, val_seq, val_lbl = split_train_val(
        train_seq, train_lbl, val_ratio=0.15
    )

    train_ds = ActionRecognitionDataset(train_seq, train_lbl)
    val_ds   = ActionRecognitionDataset(val_seq, val_lbl)
    test_ds  = ActionRecognitionDataset(test_seq, test_lbl)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False
    )

    print("=" * 60)
    print("Train dataset size:", len(train_loader.dataset))
    x, y = next(iter(train_loader))
    print("Example train batch shape:", x.shape, y.shape)

    print("Val dataset size:", len(val_loader.dataset))
    x, y = next(iter(val_loader))
    print("Example val batch shape:", x.shape, y.shape)

    print("Test dataset size:", len(test_loader.dataset))
    x, y = next(iter(test_loader))
    print("Example test batch shape:", x.shape, y.shape)

