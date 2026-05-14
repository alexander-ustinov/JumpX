"""
Проверяет, что ручной проход по слоям совпадает с выводом llm.model().
Сравнение делается по скрытым состояниям после norm, до lm_head.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16
SEQ_LEN = 128


def make_causal_mask(seq_len, device, dtype):
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]


def compute_position_embeddings(text_model, ref_tensor):
    B, L = ref_tensor.shape[:2]
    pos_ids = torch.arange(L, device=ref_tensor.device).unsqueeze(0).expand(B, -1)
    return text_model.rotary_emb(ref_tensor, pos_ids)


@torch.no_grad()
def manual_forward(text_model, input_ids):
    embeds = text_model.embed_tokens(input_ids)
    seq_len = embeds.shape[1]

    pos_embs = compute_position_embeddings(text_model, embeds)
    causal_mask = make_causal_mask(seq_len, embeds.device, embeds.dtype)
    layer_types = text_model.config.layer_types

    hiddens = embeds
    for idx, layer in enumerate(text_model.layers):
        layer_mask = None if layer_types[idx] == "linear_attention" else causal_mask
        hiddens = layer(
            hiddens,
            position_embeddings=pos_embs,
            attention_mask=layer_mask,
        )

    return text_model.norm(hiddens)


@torch.no_grad()
def reference_forward(text_model, input_ids):
    return text_model(input_ids).last_hidden_state


def main():
    print(f"Loading {MODEL_NAME} ...")
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    text_model = llm.model

    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :SEQ_LEN].to(DEVICE)
    print(f"input_ids shape: {input_ids.shape}")

    manual = manual_forward(text_model, input_ids)
    reference = reference_forward(text_model, input_ids)

    max_diff = (manual - reference).abs().max().item()
    mean_diff = (manual - reference).abs().mean().item()
    match = torch.allclose(manual, reference, atol=1e-2)

    print(f"\nmanual shape   : {manual.shape}")
    print(f"reference shape: {reference.shape}")
    print(f"\nmax  |diff|  : {max_diff:.6e}")
    print(f"mean |diff|  : {mean_diff:.6e}")
    print(f"allclose (atol=1e-2): {match}")

    if match:
        print("\n[OK] Форварды совпадают.")
    else:
        print("\n[FAIL]")


if __name__ == "__main__":
    main()
