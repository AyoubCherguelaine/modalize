import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--out", default="/data/output")
args = p.parse_args()

print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
# replace with your real fine-tuning code (Trainer / TRL SFTTrainer ...)
print("would fine-tune", args.model, "->", args.out)
