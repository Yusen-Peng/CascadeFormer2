import torch
from typing import Any, Dict
from finetuning import load_T1, load_T2, load_cross_attn_with_ffn, GaitRecognitionHeadMLP
from .constants import NUM_JOINTS_NTU, dataset_config, model_config

class CascadeFormerWrapper:
    def __init__(self, device="cuda"):
        self.device = device

        self.t1 = load_T1(model_config["t1_ckpt"], d_model=model_config["hidden_size"], num_joints=NUM_JOINTS_NTU, three_d=True, nhead=model_config["n_heads"], num_layers=model_config["num_layers"], device=device)

        self.t2 = load_T2(model_config["t2_ckpt"], d_model=model_config["hidden_size"], nhead=model_config["n_heads"], num_layers=model_config["num_layers"], device=device)
        self.cross_attn = load_cross_attn_with_ffn(model_config["cross_attn_ckpt"], d_model=model_config["hidden_size"], device=device)

        # load the gait recognition head
        self.gait_head = GaitRecognitionHeadMLP(input_dim=model_config["hidden_size"], num_classes=dataset_config["num_classes"])
        self.gait_head.load_state_dict(torch.load(model_config["gait_head_ckpt"], map_location="cpu"))
        self.gait_head = self.gait_head.to(device)

        # set models to evaluation mode
        self.t1.eval()
        self.t2.eval()
        self.cross_attn.eval()
        self.gait_head.eval()


    @torch.inference_mode()
    def infer(self, skel_batch: torch.Tensor) -> Dict[str, Any]:
        """
        skel_batch: (B, T, J, C) float32
        returns dict with logits, probs, embedding
        """
        x1 = self.t1.encode(skel_batch.to(self.device))        
        x2 = self.t2.encode(x1)
        fused = self.cross_attn(x1, x2, x2)
        pooled = fused.mean(dim=1)
        logits = self.gait_head(pooled)
        probs = torch.softmax(logits, dim=-1).float()
        embedding = torch.nn.functional.normalize(pooled, dim=-1)
        return {
            "logits": logits.cpu().numpy(),
            "probs": probs.cpu().numpy(),
            "embedding": embedding.detach().cpu().numpy(),
        }
