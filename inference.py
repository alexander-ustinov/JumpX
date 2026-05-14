"""
Minimal JumpX inference example.

Usage:
    python inference.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import JumpQwen


CKPT_PATH = "checkpoints/step_5000_17_21.pt"
START_LAYER = 17
END_LAYER = 21
MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
MAX_NEW_TOKENS = 64

PROMPTS = [
    "The capital of France is",
    "In 1969, the first human landed on the",
    "The theory of relativity was proposed by",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()

    for p in llm.parameters():
        p.requires_grad_(False)

    print(f"Loading JumpX from: {CKPT_PATH}")
    jump_model = JumpQwen.from_checkpoint(
        CKPT_PATH, START_LAYER, END_LAYER,
        base_model=llm.model,
        lm_head=llm.lm_head,
        device=device,
    )
    jump_model.eval()
    print(f"  target_pairs = {jump_model.target_pairs}  (skipping layers {START_LAYER}..{END_LAYER-1})")

    print("\n" + "=" * 60)
    for prompt in PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        generated_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                logits = jump_model(generated_ids)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_id], dim=1)
                if next_id.item() == tokenizer.eos_token_id:
                    break

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"PROMPT: {prompt}")
        print(f"OUTPUT: {output}")
        print("-" * 60)


if __name__ == "__main__":
    main()
