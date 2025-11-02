from typing import Any, Dict, List
import json
import torch
from torch.utils.data import DataLoader, Subset
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
from NTU_feeder import Feeder
from .statistics import DistanceScorer, score_anomaly, perceive_window
from .perceiver import CascadeFormerWrapper
from .rag import retrieve_context
from .constants import policy_chain, DATA_PATH, WINDOW_SIZE
from .data_utils import select_main_person_batch, extract_state_features
from .runner import is_abnormal_label
from .reinforcement import PolicyParams, decide_with_rl_policy


def decide_without_log(policies_store, incidents_store, event: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same as decide(), but does NOT modify the incidents_store or append new entries.
    Useful for offline evaluation (no side effects).
    """
    ctx, _ = retrieve_context(policies_store, incidents_store, event, scores)
    out = policy_chain.invoke({"event": event, "scores": scores, "context": ctx})

    try:
        # Remove possible Markdown code fences (```json ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip(), flags=re.MULTILINE)
        decision = json.loads(cleaned)
    except Exception as e:
        # Silent fallback to "LOG" to keep evaluation consistent
        decision = {"action": "LOG", "rationale": f"fallback: {str(e)}"}
    return decision


def process_window_without_log(policies_store, incidents_store, knn: DistanceScorer, model: CascadeFormerWrapper, skel_window: List[List[List[float]]]) -> Dict[str, Any]:
    # 1) Perceive
    event = perceive_window.invoke({"model": model, "skel_window": skel_window})

    # 2) Score anomaly
    scores = score_anomaly.invoke({"knn": knn, "event": event})

    # 3) Decide (your normal function)
    decision = decide_without_log(policies_store, incidents_store, event, scores)

    return {"event": event, "scores": scores, "decision": decision}

def evaluate_full_test_split_with_agent(policies_store, incidents_store, knn_scorer, model: CascadeFormerWrapper,
                                        batch_size: int = 16, device: str = "cuda"):
    """
    Loops over the test split and calls your existing process_window(...) per sample.
    Metrics are computed for:
      y_true: 1 if GT label ∈ abnormal_action_labels else 0
      y_pred: 1 if decision['action'] == 'ALERT' else 0
    """
    test_dataset = Feeder(
        data_path=DATA_PATH,
        split='test',
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
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    model.t1.eval()
    model.t2.eval()
    model.cross_attn.eval()
    model.gait_head.eval()

    with torch.inference_mode():
        for skeletons, labels, _ in tqdm(loader):
            skeletons = skeletons.to(device)                 # (B,C,T,V,M)
            labels_np = labels.cpu().numpy().astype(int)     # (B,)
            windows = select_main_person_batch(skeletons)   # (B,T,V,C)

            # Iterate samples in this batch and reuse the full agent path
            for i in range(windows.shape[0]):
                window_np = windows[i].cpu().numpy().astype(float)  # (T,V,C)
                result = process_window_without_log(
                    policies_store, incidents_store,
                    knn=knn_scorer, model=model,
                    skel_window=window_np.tolist()
                )

                pred_alert = 1 if str(result["decision"]["action"]).upper() == "ALERT" else 0
                true_abn   = 1 if is_abnormal_label(int(labels_np[i])) else 0

                y_pred.append(pred_alert)
                y_true.append(true_abn)
            
            break # for quick demo, remove this line to run full test split

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    print("\n=== Offline Evaluation (Agent over TEST split) ===", flush=True)
    print(f"Samples   : {len(y_true)}", flush=True)
    print(f"Accuracy  : {acc:.4f}", flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"F1-score  : {f1:.4f}", flush=True)
    print("Confusion Matrix (rows=truth [Normal, Abnormal]; cols=pred [LOG, ALERT])", flush=True)
    print(cm, flush=True)


def evaluate_random_batches_with_agent(policies_store, incidents_store, knn_scorer, model: CascadeFormerWrapper,
                                       num_batches: int = 10, batch_size: int = 16, device: str = "cuda"):
    """
    Randomly samples `num_batches` batches from the test split (each of size `batch_size`).
    Runs the full agent inference path and computes classification metrics.
    """
    test_dataset = Feeder(
        data_path=DATA_PATH,
        split='test',
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

    y_true, y_pred = [], []

    model.t1.eval()
    model.t2.eval()
    model.cross_attn.eval()
    model.gait_head.eval()

    rng = np.random.default_rng(42)
    N = len(test_dataset)
    k = min(num_batches * batch_size, N)   # sample k items (not batches)

    idx = rng.choice(N, size=k, replace=False)  # unique item indices
    subset = Subset(test_dataset, idx.tolist())

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,  # no need to shuffle: indices are already random
    )

    with torch.inference_mode():
        for skeletons, labels, _ in tqdm(
            loader, 
            desc="Evaluating random batches"
        ):
            skeletons = skeletons.to(device)                 
            labels_np = labels.cpu().numpy().astype(int)     
            windows = select_main_person_batch(skeletons)   # (B,T,V,C)

            for i in range(windows.shape[0]):
                window_np = windows[i].cpu().numpy().astype(float)
                result = process_window_without_log(
                    policies_store, incidents_store,
                    knn=knn_scorer, model=model,
                    skel_window=window_np.tolist()
                )

                pred_alert = 1 if str(result["decision"]["action"]).upper() == "ALERT" else 0
                true_abn   = 1 if is_abnormal_label(int(labels_np[i])) else 0

                y_pred.append(pred_alert)
                y_true.append(true_abn)

    # --- compute metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    print("\n=== Offline Evaluation (Agent over TEST split) ===", flush=True)
    print(f"Samples   : {len(y_true)}", flush=True)
    print(f"Accuracy  : {acc:.4f}", flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"F1-score  : {f1:.4f}", flush=True)
    print("Confusion Matrix (rows=truth [Normal, Abnormal]; cols=pred [LOG, ALERT])", flush=True)
    print(cm, flush=True)


def process_window_no_RAG(
    knn: DistanceScorer, model: CascadeFormerWrapper,
    skel_window: List[List[List[float]]],
    learned_params: PolicyParams,
) -> Dict[str, Any]:
    """
    Hybrid inference that combines KB-based decision and RL-based decision.
    If prefer_kb=True, KB decision takes precedence on disagreement (especially for ALERT cases).
    """
    # 1) Perceive
    event = perceive_window.invoke({"model": model, "skel_window": skel_window})

    # 2) Score anomaly (must provide knn_dist / mahalanobis / top1_conf)
    scores = score_anomaly.invoke({"knn": knn, "event": event})

    # RL decision
    s = extract_state_features(event, scores)
    rl_action = decide_with_rl_policy(s, learned_params)["action"].upper()

    action = rl_action
    decision = {"action": action, "rationale": ""}
    return {"event": event, "scores": scores, "decision": decision}


def evaluate_full_test_split_policy_only(
    knn_scorer,
    model: CascadeFormerWrapper,
    learned_params: PolicyParams,
    batch_size,
    device: str = "cuda"
):
    """
    Randomly samples `num_batches` batches from the test split.
    Runs the RL policy on each sample and computes classification metrics.
    """
    test_dataset = Feeder(
        data_path=DATA_PATH,
        split="test",
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=WINDOW_SIZE,
        normalization=False,
        random_rot=False,
        p_interval=[0.5, 1],
        vel=False,
        bone=False,
    )
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # --- 3. Metric containers ---
    y_true, y_pred = [], []

    # --- 4. Set model to eval mode ---
    model.t1.eval()
    model.t2.eval()
    model.cross_attn.eval()
    model.gait_head.eval()

    # --- 5. Evaluation loop ---
    with torch.inference_mode():
        for skeletons, labels, _ in tqdm(loader):


            skeletons = skeletons.to(device)
            labels_np = labels.cpu().numpy().astype(int)
            windows = select_main_person_batch(skeletons)

            for i in range(windows.shape[0]):
                window_np = windows[i].cpu().numpy().astype(float)
                result = process_window_no_RAG(
                    knn=knn_scorer,
                    model=model,
                    skel_window=window_np.tolist(),
                    learned_params=learned_params,
                )
                pred_alert = 1 if str(result["decision"]["action"]).upper() == "ALERT" else 0
                true_abn = 1 if is_abnormal_label(int(labels_np[i])) else 0
                y_pred.append(pred_alert)
                y_true.append(true_abn)

    # --- 6. Compute metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # --- 7. Print results ---
    print("\n=== Offline Evaluation (evaluating Policy over full TEST Batches) ===", flush=True)
    print(f"Samples   : {len(y_true)}", flush=True)
    print(f"Accuracy  : {acc:.4f}", flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"F1-score  : {f1:.4f}", flush=True)

