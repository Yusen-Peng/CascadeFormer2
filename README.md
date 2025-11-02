# 🎁 CascadeFormer 2: Towards a Cost-Aligned Anomaly Detection Agent via Reinforcement Learning with Verifiable Rewards

## Agent Design

![alt text](/CascadeFormer-AD-Agent.png)

## Commands

```bash
# export the API key first
export OPENAI_API_KEY=<API KEY GOES HERE>
# log incidents with statistics (optional) + inference demo
CUDA_VISIBLE_DEVICES=0 taskset -c 20-30 python baseline/action_recognition/cascadeformer_1_0/joint/ntu_60_own/agent_demo.py
# reinforcement-learning
CUDA_VISIBLE_DEVICES=0 taskset -c 20-30 python baseline/action_recognition/cascadeformer_1_0/joint/ntu_60_own/agent_RL.py
# evaluation
CUDA_VISIBLE_DEVICES=0 taskset -c 20-30 python baseline/action_recognition/cascadeformer_1_0/joint/ntu_60_own/agent_eval.py
```
