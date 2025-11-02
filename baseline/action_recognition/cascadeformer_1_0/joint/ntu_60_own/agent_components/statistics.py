from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
from typing import Any, Dict, List
from langchain_core.tools import tool
from NTU_feeder import Feeder
from .perceiver import CascadeFormerWrapper
from .normal_bank import build_normal_bank
from .constants import classify_labels

class DistanceScorer:
    """
        How far is this embedding z from normal training data?
    """
    def __init__(self, model: CascadeFormerWrapper, k=5):
        self.model = model
        self.k = k

        WINDOW_SIZE = 64

        DATA_PATH = "NTU60_CS.npz"
        train_dataset = Feeder(
            data_path=DATA_PATH,
            split='train',
            debug=False,
            random_choose=False,
            random_shift=False,
            random_move=False,
            window_size=WINDOW_SIZE,
            normalization=False,
            random_rot=True,
            p_interval=[0.5, 1],
            vel=False,
            bone=False
        )

        normal_bank = build_normal_bank(self.model, train_dataset, per_class=False)
        self.nn = NearestNeighbors(n_neighbors=k).fit(normal_bank)
        self.normal_bank = normal_bank

    def score(self, z: np.ndarray) -> float:
        dists, _ = self.nn.kneighbors(z.reshape(1, -1))
        return float(dists.mean())


def entropy(p: np.ndarray) -> float:
    """
        How uncertain is the model's prediction?
    """
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())


def fit_mahalanobis_params(normal_bank: np.ndarray, reg: float = 1e-5):
    """
    From a bank of normal embeddings (N, D), compute the mean and (regularized) inverse covariance.
    Returns (mean, inv_cov).
    """
    if normal_bank.ndim != 2:
        raise ValueError("normal_bank must be a 2D array of shape (N, D).")
    mu = normal_bank.mean(axis=0)
    # Sample covariance (D, D)
    cov = np.cov(normal_bank, rowvar=False)
    # Tikhonov regularization for stability
    inv_cov = np.linalg.pinv(cov + reg * np.eye(cov.shape[0], dtype=cov.dtype))
    return mu.astype(np.float32), inv_cov.astype(np.float32)

def mahalanobis_distance(z: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Mahalanobis distance of embedding z to a Gaussian fit (mean, inv_cov) from normal data.
    """
    z = z.astype(np.float32)
    mean = mean.astype(np.float32)
    inv_cov = inv_cov.astype(np.float32)
    diff = z - mean
    # sqrt( (z - μ)^T Σ^{-1} (z - μ) )
    return float(np.sqrt(diff @ inv_cov @ diff))


@tool("perceive_window", return_direct=False)
def perceive_window(model: CascadeFormerWrapper, skel_window: List[List[List[float]]]) -> Dict[str, Any]:
    """
        run CascadeFormer on a single window of skeletons and return structured event with probs, entropy, and embedding.
    """
    x = torch.tensor(skel_window, dtype=torch.float32).unsqueeze(0) # shape: (1,T,J,C)
    out = model.infer(x)
    probs = out["probs"][0]
    embedding = out["embedding"]

    event = {
        "top_label": classify_labels[int(np.argmax(probs))],
        "top_prob": float(np.max(probs)),
        "entropy": entropy(probs),
        "embedding": embedding[0].tolist(),  # convert to list for JSON serialization
    }
    return event

@tool("score_anomaly", return_direct=False)
def score_anomaly(knn: DistanceScorer, event: Dict[str, Any]) -> Dict[str, Any]:
    """
        compute anomaly scores from embedding and additional signals.
    """
    z = np.array(event["embedding"], dtype=np.float32)
    # compute Mahalanobis distance
    mu, inv_cov = fit_mahalanobis_params(knn.normal_bank)

    scores = {
        "knn_dist": knn.score(z),
        "mahalanobis": mahalanobis_distance(z, mu, inv_cov),
        "ent": event["entropy"],
        "top1_conf": event["top_prob"],
    }
    return scores