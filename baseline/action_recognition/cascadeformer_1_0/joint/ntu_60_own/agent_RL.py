import os
from typing import List, Dict, Any
import joblib
import pandas as pd
import matplotlib
from pathlib import Path
matplotlib.use("Agg")
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from agent_components.constants import ST_RE
from agent_components.perceiver import CascadeFormerWrapper
from agent_components.statistics import DistanceScorer, score_anomaly, perceive_window
from agent_components.rag import print_incident_db, print_policy_db, write_policy_db
from agent_components.reinforcement import PolicyParams, train_a_reward_model, policy_search, decide_with_rl_policy, fixed_threshold_policy_search, random_threshold_policy_search
from agent_components.data_utils import extract_state_features
from agent_components.visualization import visualize_knn_maha_scatter


# Load environment variables from .env file
load_dotenv(dotenv_path="/home/peng.1007/CascadeFormer/.env")
# Access your key
api_key = os.getenv("OPENAI_KEY")


def process_window_RL(
    policies_store, incidents_store,
    knn: DistanceScorer, model: CascadeFormerWrapper,
    skel_window: List[List[List[float]]],
    learned_params: PolicyParams,
    prefer_kb: bool = False,   # ✅ True = favor KB when KB and RL disagree
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

def rl_policy_optimization(incidents_df: pd.DataFrame,
                                policies_store: FAISS, incidents_store: FAISS,
                                knn_scorer, model: CascadeFormerWrapper,
                                device: str = "cuda", search_mode: str = "RLVR") -> FAISS:
    # 1) Train reward model from past incidents
    r_model = train_a_reward_model(incidents_df)

    # 2) Search best policy params under learned R(s,a) reward model


    if search_mode == "fixed":
        best_params: PolicyParams = fixed_threshold_policy_search(incidents_df, knn_quantile=0.90, maha_quantile=0.90)
        print("\n=== Fixed Quantile-based Policy Optimization Result ===", flush=True)
        print("[Fixed] Best params:", best_params)
        print("===========================================", flush=True)
    elif search_mode == "random":
        best_params: PolicyParams = random_threshold_policy_search(incidents_df, rng=67)
        print("\n=== Random Threshold-based Policy Optimization Result ===", flush=True)
        print("[Random] Best params:", best_params)
        print("===========================================", flush=True)
    elif search_mode == "RLVR":
        best_params: PolicyParams = policy_search(r_model, incidents_df)
        print("\n=== RL-based Policy Optimization Result ===", flush=True)
        print("[RLVR] Best params:", best_params)
        print("===========================================", flush=True)
    elif search_mode == "llm":
        raise NotImplementedError("LLM-based policy optimization is not implemented yet.")
    else:
        raise ValueError(f"Unknown search_mode: {search_mode}")

    # 3) Rebuild the policy part of the knowledge base with the learned params
    policy_text = (
        f"Raise an ALERT if BOTH of the following conditions are met:\n"
        # f"- entropy >= {best_params.max_entropy:.4f}\n"
        f"- knn_dist >= {best_params.min_knn:.4f}\n"
        f"- mahalanobis >= {best_params.min_maha:.4f}\n"
        # f"- (1 - top1_conf) >= {best_params.min_low_conf:.4f}\n"
        f"Otherwise, LOG the event as normal."
    )
    
    # 4) return a new policy store
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    new_policies_store = FAISS.from_texts(
        texts=[policy_text],
        embedding=emb
    )

    return new_policies_store


def agent_rl_policy_optimization(search_mode: str):
    """
    RL-based policy optimization entrypoint.
    Builds incidents_df from KB, trains reward model, searches best policy.
    """
    model = CascadeFormerWrapper(device="cuda")
    
    # ---- Load or build KNN scorer ----
    if not os.path.exists("trained_knn.pkl"):
        knn = DistanceScorer(model=model)
        joblib.dump(knn, "trained_knn.pkl")
    knn = joblib.load("trained_knn.pkl")

    # ---- Load vector stores ----
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    policies_store  = FAISS.load_local("vectorstores/initial_policies",  emb, allow_dangerous_deserialization=True)
    incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)

    print_incident_db(incidents_store)
    print_policy_db(policies_store)

    def _parse_incidents_kb_file(path: str = "incidents_db.kb") -> pd.DataFrame:
        p = Path(path)
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("DUMMY") or set(s) == {"-"}:
                continue
            m = ST_RE.search(s)
            d = m.groupdict()
            rows.append({
                "entropy": float(d["entropy"]),
                "knn_dist": float(d["knn_dist"]),
                "mahalanobis": float(d["mahalanobis"]),
                "top1_conf": float(d["top1_conf"]),
                "decision": d["decision"].upper(),
                "gt_label": (d["gt_label"]).lower(),
            })
        return pd.DataFrame(rows)

    incidents_df = _parse_incidents_kb_file("incidents_db.kb")
    print(f"[incidents_df] {len(incidents_df)} rows | columns: {list(incidents_df.columns)}")

    new_policy_store = rl_policy_optimization(
        incidents_df,
        policies_store, incidents_store,
        knn, model,
        device="cuda",
        search_mode=search_mode
    )

    # RL-based policies in vectorstore
    new_policy_store.save_local("vectorstores/rl_learned_policies")
    # RL-based policies in text
    write_policy_db(new_policy_store, path="rl_learned_policies.kb")
    print("new policy is out: " , flush=True)
    print("😛" * 20, flush=True)
    print_policy_db(new_policy_store)


def visualization():
    def _parse_incidents_kb_file(path: str = "incidents_db.kb") -> pd.DataFrame:
        p = Path(path)
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("DUMMY") or set(s) == {"-"}:
                continue
            m = ST_RE.search(s)
            d = m.groupdict()
            rows.append({
                "entropy": float(d["entropy"]),
                "knn_dist": float(d["knn_dist"]),
                "mahalanobis": float(d["mahalanobis"]),
                "top1_conf": float(d["top1_conf"]),
                "decision": d["decision"].upper(),
                "gt_label": (d["gt_label"]).lower(),
            })
        return pd.DataFrame(rows)

    incidents_df = _parse_incidents_kb_file("incidents_db.kb")
    print(f"[incidents_df] {len(incidents_df)} rows | columns: {list(incidents_df.columns)}")

    fig_path = visualize_knn_maha_scatter(
        incidents_df,
        outpath="knn_maha_scatter.png",
        log_maha=True  # toggle if your maha scale is wide
    )
    print(f"[viz] Saved KNN vs. Mahalanobis scatter to {fig_path}", flush=True)

if __name__ == "__main__":
    search_mode = "fixed" # options: "RLVR", "fixed", "random"
    agent_rl_policy_optimization(search_mode=search_mode)
    #visualization()



