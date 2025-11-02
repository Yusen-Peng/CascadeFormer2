from typing import Dict, Union
from langchain_community.vectorstores import FAISS


def retrieve_context(policy_only: bool, policies_store: FAISS, incidents_store: FAISS, event: Dict[str, Union[str, float]], scores: Dict[str, float]):
    """
        retrieve context from the knowledge base.
    """

    q = f"knn_dist={scores['knn_dist']:.4f} mahalanobis={scores['mahalanobis']:.4f}"
    pol = "\n".join([d.page_content for d in policies_store.similarity_search(q, k=3)])
    if policy_only:
        return f"[POLICIES]\n{pol}", q
    else:
        inc = "\n".join([d.page_content for d in incidents_store.similarity_search(q, k=3)])
        return f"[POLICIES]\n{pol}\n\n[SIMILAR INCIDENTS]\n{inc}", q


def print_incident_db(incidents_store: FAISS):
    print("\n=== Incidents in the knowledge base ===", flush=True)
    for _, doc in incidents_store.docstore._dict.items():
        print(doc.page_content, flush=True)
        print("-------------------", flush=True)

def write_incident_db(incidents_store: FAISS):
    with open("incidents_db.kb", "w") as f:
        for _, doc in incidents_store.docstore._dict.items():
            f.write(doc.page_content + "\n")
            f.write("-------------------\n")


def print_policy_db(policies_store: FAISS):
    print("\n=== Policies in the knowledge base ===", flush=True)
    for _, doc in policies_store.docstore._dict.items():
        print(doc.page_content, flush=True)
        print("-------------------", flush=True)

def write_policy_db(policies_store: FAISS, path: str):
    with open(path, "w") as f:
        for _, doc in policies_store.docstore._dict.items():
            f.write(doc.page_content + "\n")
            f.write("-------------------\n")