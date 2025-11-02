from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import re

WINDOW_SIZE = 64
DATA_PATH = "NTU60_CS.npz"

NTU25_EDGES = [
    (0, 1), (1, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (20, 1), (21, 7),
    (22, 7), (23, 11), (24, 11)
]

NUM_JOINTS_NTU = 25

dataset_config = {"num_classes": 60}

classify_labels = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", 
    "drop", "pickup", "throw", "sitting down", "standing up (from sitting position)", 
    "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", 
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", 
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping (one foot jumping)", "jump up", 
    "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", 
    "pointing to something with finger", "taking a selfie", "check time (from watch)", 
    "rub two hands together", "nod head/bow", "shake head", "wipe face", 
    "salute", "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "staggering", "falling", "touch head (headache)", 
    "touch chest (stomachache/heart pain)", "touch back (backache)", 
    "touch neck (neckache)",  "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "hugging other person",
    "giving something to other person",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
    "point finger at the other person",
    "touch other person's pocket"
]


normal_action_labels = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair",
    "drop", "pickup", "throw", "sitting down", "standing up (from sitting position)",
    "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping (one foot jumping)", "jump up",
    "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie", "check time (from watch)",
    "rub two hands together", "nod head/bow", "shake head", "wipe face",
    "salute", "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "use a fan (with hand or paper)/feeling warm",
    "pat on back of other person", "hugging other person", "giving something to other person",
    "handshaking", "walking towards each other", "walking apart from each other"
]

abnormal_action_labels = [
    "staggering", "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea or vomiting condition",
    "punching/slapping other person", "kicking other person",
    "pushing other person", "point finger at the other person", "touch other person's pocket"
]


model_config = {
    "t1_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_T1.pt",
    "t2_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_T2.pt",
    "cross_attn_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_cross_attn.pt",
    "gait_head_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_head.pt",
    "hidden_size": 768, # for NTU/CS
    "n_heads": 16,  # for NTU/CS
    "num_layers": 16,  # for NTU/CS
}

POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a surveillance policy agent. You receive a perception event JSON, anomaly scores, "
     "and retrieved policies/incidents. Decide ONE action: LOG or ALERT."
     "Raise an ALERT if the observed action is abnormal; Log the event if the observed action is normal."
     "Return JSON with keys: action, rationale."),
    ("human", "Event:\n{event}\n\nScores:\n{scores}\n\nContext:\n{context}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
policy_chain = POLICY_PROMPT | llm | StrOutputParser()

WEIGHTS = {"tp": +5, "fp": -3, "fn": -10, "tn": 0}

#SEARCH_BARS = [0.55, 0.65, 0.74, 0.76, 0.78, 0.82, 0.84, 0.86, 0.87, 0.88, 0.885, 0.89, 0.895]

SEARCH_BARS = [0.76, 0.78, 0.82, 0.84, 0.86, 0.87, 0.88, 0.885, 0.89, 0.895]


ST_RE = re.compile(
    r"\[statistics\]:\s*"
    r"entropy=(?P<entropy>[+-]?\d+(?:\.\d+)?)\s+"
    r"knn_dist=(?P<knn_dist>[+-]?\d+(?:\.\d+)?)\s+"
    r"mahalanobis=(?P<mahalanobis>[+-]?\d+(?:\.\d+)?)\s+"
    r"top1_conf=(?P<top1_conf>[+-]?\d+(?:\.\d+)?)\s*-\s*"
    r"\[decision\]:(?P<decision>ALERT|LOG)"
    r"(?:\s*\|\s*\[ground_truth\]:(?P<gt_label>\w+))?\s*$"
)