"""
Baseline eval: полная модель (без прыжков) на MMLU + QuALITY.
Запуск: torchrun --nproc_per_node=8 eval_baseline.py
"""
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import json
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from X.eval.utils import prepare_mmlu
from X.eval.quality_loader import load_quality
from eval.eval_utils import make_quality_prompt

MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
DTYPE = "bfloat16"

MMLU_N = 2000
QUALITY_N = 2000
QUALITY_MAX_LEN = 5000

LABEL_LIST = ["A", "B", "C", "D"]


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    return rank, world_size, device


@torch.no_grad()
def eval_mmlu(model, tokenizer, letter_ids, rank, world_size, device):
    data = prepare_mmlu(tokenizer, MMLU_N)
    shard = data[rank::world_size]

    local_correct = 0

    for example in tqdm(shard, desc="MMLU baseline", disable=rank != 0):
        input_ids = example["input_ids"].unsqueeze(0).to(device)
        label_letter = example["label_letter"]

        logits = model(input_ids=input_ids, use_cache=False).logits
        candidate_logits = logits[0, -1, letter_ids].float()
        pred_letter = LABEL_LIST[candidate_logits.argmax().item()]

        if pred_letter == label_letter:
            local_correct += 1

    correct_t = torch.tensor([local_correct], dtype=torch.long, device=device)
    dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)

    return {"accuracy": correct_t.item() / len(data)}


@torch.no_grad()
def eval_quality(model, tokenizer, letter_ids, rank, world_size, device):
    ds = load_quality(html_stripped=True)
    dataset = ds["dev"].select(range(min(QUALITY_N, len(ds["dev"]))))
    n = len(dataset)
    indices = list(range(rank, n, world_size))

    local_correct = 0
    local_total = 0

    for i in tqdm(indices, desc="QuALITY baseline", disable=rank != 0):
        ex = dataset[i]

        if ex["label"] is None:
            continue

        prompt = make_quality_prompt(ex)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]

        if input_ids.shape[1] > QUALITY_MAX_LEN:
            continue

        input_ids = input_ids.to(device)
        logits = model(input_ids=input_ids, use_cache=False).logits
        pred = int(logits[0, -1, letter_ids].argmax().item())

        gold = int(ex["label"])
        local_correct += int(pred == gold)
        local_total += 1

    correct_t = torch.tensor([local_correct], dtype=torch.long, device=device)
    total_t = torch.tensor([local_total], dtype=torch.long, device=device)
    dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_t, op=dist.ReduceOp.SUM)

    total = total_t.item()
    return {"accuracy": correct_t.item() / total if total > 0 else 0.0}


def main():
    rank, world_size, device = setup()

    if rank == 0:
        print(f"Loading {MODEL_NAME} on {world_size} GPUs...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=getattr(torch, DTYPE),
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    letter_ids = [
        tokenizer.encode(" " + letter, add_special_tokens=False)[0]
        for letter in LABEL_LIST
    ]

    mmlu = eval_mmlu(model, tokenizer, letter_ids, rank, world_size, device)
    torch.cuda.empty_cache()

    quality = eval_quality(model, tokenizer, letter_ids, rank, world_size, device)
    torch.cuda.empty_cache()

    if rank == 0:
        results = {
            "model": MODEL_NAME,
            "mmlu_accuracy": mmlu["accuracy"],
            "quality_accuracy": quality["accuracy"],
            "mmlu_N": MMLU_N,
            "quality_N": QUALITY_N,
            "quality_max_len": QUALITY_MAX_LEN,
        }
        print(json.dumps(results, indent=2))

        with open("eval/baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
