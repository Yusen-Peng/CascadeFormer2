from typing import Any, Dict
import time
import csv
import re
import json
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from .constants import policy_chain

from .rag import retrieve_context

@tool("log_event", return_direct=False)
def log_event(event: Dict[str, Any], scores: Dict[str, Any]) -> str:
    """
        persist the event + scores into a CSV file.
    """
    with open("events.csv", "a", newline="") as f:
        fieldnames = list(event.keys()) + list(scores.keys()) + ["ts"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file is empty
        if f.tell() == 0:
            writer.writeheader()

        row = {**event, **scores, "ts": time.time()}
        writer.writerow(row)
    return "logged"

@tool("raise_alert", return_direct=False)
def raise_alert(event: Dict[str, Any], scores: Dict[str, Any]) -> str:
    """
        send an alert with a short rationale.
    """
    return f"ALERT: {event['top_label']} (p={event['top_prob']:.2f}) ent={event['entropy']:.2f} knn_dist={scores['knn_dist']:.2f}"


def decide(policies_store: FAISS, incidents_store: FAISS, event: Dict[str, Any], scores: Dict[str, Any], gt_label: str) -> Dict[str, Any]:
    """
    make a decision based on the event, scores, and context.
    """
    ctx, q = retrieve_context(policies_store, incidents_store, event, scores)
    out = policy_chain.invoke({"event": event, "scores": scores, "context": ctx})
    # Remove Markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip(), flags=re.MULTILINE)

    decision = json.loads(cleaned)
    # insert the context into the knowledge base
    incident = f"[statistics]:{q} - [decision]:{decision['action']} | [ground_truth]:{gt_label}"
    incidents_store.add_texts([incident], metadatas=[{"kind": "incident_context"}])
    return decision
