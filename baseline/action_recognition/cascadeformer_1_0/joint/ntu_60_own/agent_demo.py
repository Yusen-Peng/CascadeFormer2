import os
import joblib
import matplotlib
matplotlib.use("Agg")
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from agent_components.constants import abnormal_action_labels
from agent_components.perceiver import CascadeFormerWrapper
from agent_components.statistics import DistanceScorer
from agent_components.rag import write_incident_db, print_incident_db, write_policy_db, print_policy_db
from agent_components.runner import train_one_sample, inference_demo


# Load environment variables from .env file
load_dotenv(dotenv_path="/home/peng.1007/CascadeFormer/.env")
# Access your key
api_key = os.getenv("OPENAI_KEY")


def agent_training_and_demo(inference_only: bool):
    INFERENCE_ONLY = inference_only
    n_samples = 10_000
    model = CascadeFormerWrapper(device="cuda")
    
    # if no trained knn, create one
    if not os.path.exists("trained_knn.pkl"):
        knn = DistanceScorer(model=model)
        # save KNN model for later use
        joblib.dump(knn, "trained_knn.pkl")

    # load the KNN
    knn = joblib.load("trained_knn.pkl")
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    if not INFERENCE_ONLY:
        # build new vector stores
        policies_store = FAISS.from_texts(
            texts=[
                f"Raise an ALERT if the predicted action is within the abnormal action list "f"({', '.join(abnormal_action_labels)})."
            ],
            embedding=emb
        )

        incidents_store = FAISS.from_texts(
            texts=[
                "DUMMY INCIDENT ENTRY; DO NOT USE.",
            ],
            embedding=emb
        )
    else:
        # load existing vector stores
        policies_store = FAISS.load_local("vectorstores/initial_policies", emb, allow_dangerous_deserialization=True)
        incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)


    if not INFERENCE_ONLY:
        # training code
        print(f"Start training with {n_samples} samples...", flush=True)
        for i in range(n_samples):
            print(f"\n=== Training iteration {i+1}/{n_samples} ===", flush=True)
            train_one_sample(policies_store, incidents_store, model, knn, json_path="demo_window.json")

        os.makedirs("vectorstores", exist_ok=True)
        policies_store.save_local("vectorstores/initial_policies")
        incidents_store.save_local("vectorstores/incidents")

        # write the knowledge base to text files
        # NOTE: IMPORTANT - only write when training, not during inference
        write_incident_db(incidents_store)
        write_policy_db(policies_store, path="initial_policies_db.kb")


    # print both policies and incidents
    print_incident_db(incidents_store)
    print_policy_db(policies_store)

    print("\n\nStart an inference demo...", flush=True)
    DEMO_VIDEO_PATH = "demo_window.mp4"
    DEMO_JSON_PATH = "demo_window.json"
    policies_store = FAISS.load_local("vectorstores/initial_policies", emb, allow_dangerous_deserialization=True)
    incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)

    inference_demo(policies_store, incidents_store, model, knn, json_path=DEMO_JSON_PATH, video_path=DEMO_VIDEO_PATH)


if __name__ == "__main__":
    agent_training_and_demo(inference_only=False)
