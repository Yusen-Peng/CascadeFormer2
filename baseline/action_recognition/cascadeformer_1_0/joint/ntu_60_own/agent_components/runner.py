import json
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from .statistics import DistanceScorer, score_anomaly, perceive_window
from .decision import decide, log_event, raise_alert
from .perceiver import CascadeFormerWrapper
from .demo_utils import make_skeleton_video
from .data_utils import prepare_one_sample
from .constants import classify_labels, abnormal_action_labels


def is_abnormal_label(label_id: int) -> bool:
    return classify_labels[label_id] in set(abnormal_action_labels)


def process_window(policies_store: FAISS, incidents_store: FAISS, knn: DistanceScorer, model: CascadeFormerWrapper, skel_window: List[List[List[float]]], gt_label: str) -> Dict[str, Any]:
    # 1) Perceive
    event = perceive_window.invoke({"model": model, "skel_window": skel_window})

    # 2) Score anomaly
    scores = score_anomaly.invoke({"knn": knn, "event": event})

    # 3) Decide (your normal function)
    decision = decide(policies_store, incidents_store, event, scores, gt_label)

    # 4) Act/log via tools
    if decision["action"] == "ALERT":
        msg = raise_alert.invoke({"event": event, "scores": scores})
    else:
        msg = log_event.invoke({"event": event, "scores": scores})

    return {"event": event, "scores": scores, "decision": decision, "result": msg}

def run_on_single_json(policies_store, incidents_store, knn: DistanceScorer, model: CascadeFormerWrapper, json_path: str, gt_label: str):
    """
    Load one skeleton window from a JSON file and run the agent once.
    JSON format: nested list shaped like (T, J, C).
    """
    with open(json_path, "r") as f:
        skel_window = json.load(f)

    out = process_window(policies_store, incidents_store, knn, model, skel_window, gt_label)

    print("=== 🔥Demo Run🔥 ===", flush=True)
    print("🔥Decision:", json.dumps(out["decision"]["action"], indent=2), flush=True)
    print("🔥Rationale:", json.dumps(out["decision"]["rationale"], indent=2), flush=True)



def inference_demo(
        policies_store,
        incidents_store,
        model: CascadeFormerWrapper, 
        knn: DistanceScorer,
        json_path="demo_window.json",
        video_path="demo_window.mp4"
    ):
    # prepare one sample json
    _, label_id, window = prepare_one_sample(json_path, shuffle=True, train=False)

    if is_abnormal_label(label_id):
        gt_label = "abnormal"
    else:
        gt_label = "normal"

    # visualize the skeleton window
    make_skeleton_video(window, out_path=video_path, fps=12, use_3d=True)

    # run the agent on this json
    run_on_single_json(policies_store, incidents_store, knn, model, json_path, gt_label)


def train_one_sample(
        policies_store,
        incidents_store,
        model: CascadeFormerWrapper, 
        knn: DistanceScorer,
        json_path="demo_window.json"
    ):
    # prepare one sample json
    _, label_id, _ = prepare_one_sample(json_path, shuffle=True, train=True)

    if is_abnormal_label(label_id):
        gt_label = "abnormal"
    else:
        gt_label = "normal"

    # run the agent on this json
    run_on_single_json(policies_store, incidents_store, knn, model, json_path, gt_label)