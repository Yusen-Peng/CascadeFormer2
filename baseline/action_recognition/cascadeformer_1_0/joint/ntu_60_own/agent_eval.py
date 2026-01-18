import os
import joblib
import matplotlib
matplotlib.use("Agg")
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agent_components.perceiver import CascadeFormerWrapper
from agent_components.statistics import DistanceScorer
from agent_components.rag import print_incident_db, print_policy_db
from agent_components.eval import evaluate_full_test_split_with_agent, evaluate_random_batches_with_agent, evaluate_full_test_split_policy_only
from agent_components.reinforcement import PolicyParams

def main():
    MODE = "random" # 'full' or 'random' or 'policy_only'
    POLICY = "RL" # 'RL' 'classification'



    model = CascadeFormerWrapper(device="cuda")
    # ---- Load or build KNN scorer ----
    if not os.path.exists("trained_knn.pkl"):
        knn = DistanceScorer(model=model)
        joblib.dump(knn, "trained_knn.pkl")
    knn_scorer = joblib.load("trained_knn.pkl")

    # ---- Load vector stores ----
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)
    
    if POLICY == "classification":
        policies_store  = FAISS.load_local("vectorstores/initial_policies",  emb, allow_dangerous_deserialization=True)
    elif POLICY == "RL":
        policies_store  = FAISS.load_local("vectorstores/rl_learned_policies",  emb, allow_dangerous_deserialization=True)
    else:
        raise ValueError(f"Unknown POLICY: {POLICY}")
    
    print_incident_db(incidents_store)
    print_policy_db(policies_store)

    if MODE == "full":
        evaluate_full_test_split_with_agent(
            policies_store,
            incidents_store,
            knn_scorer,
            model,
            batch_size=16,
            device="cuda"
        )
    elif MODE == "random":
        NUM_BATCHES = 20
        evaluate_random_batches_with_agent(
            policies_store,
            incidents_store,
            knn_scorer,
            model,
            num_batches=NUM_BATCHES,
            batch_size=16,
            device="cuda"
        )
    elif MODE == "policy_only":

        # CascadeFormerAgent (our method)
        PARAMS = PolicyParams(
            max_entropy = 0,
            min_knn = 1.0935,
            min_maha = 74.1626,
            min_low_conf = 0
        )

        # fixed threshold
        # PARAMS = PolicyParams(
        #     max_entropy = 0,
        #     min_knn = 1.1568,
        #     min_maha = 78.7561,
        #     min_low_conf = 0
        # )

        evaluate_full_test_split_policy_only(
            knn_scorer,
            model,
            PARAMS,
            batch_size=16,
            device="cuda"
        )
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
