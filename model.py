import torch
import torch.nn as nn
import torch.nn.functional as F


def make_causal_mask(
        seq_len: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor:

    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

def compute_position_embeddings(
        text_model, 
        ref_tensor, 
        past_seen_tokens=0
    ):

    B, L = ref_tensor.shape[:2]

    pos_ids = torch.arange(
        past_seen_tokens,
        past_seen_tokens + L,
        device=ref_tensor.device,
    ).unsqueeze(0).expand(B, -1)  # [B, L]

    return text_model.rotary_emb(ref_tensor, pos_ids)  # returns (cos, sin)

def pair_key(pair):
    return f"{pair[0]}_{pair[1]}"




class JumpQwen(nn.Module):

    def __init__(self, base_model, lm_head, cfg, target_pairs=None):
        super().__init__()

        self.cfg = cfg
        self.base_model = base_model
        self.layers = self.base_model.layers
        self.layer_types = self.base_model.config.layer_types
        self.target_pairs = target_pairs or [[22, 28]]
        self.starts = [x[0] for x in self.target_pairs]
        self.ends = [x[1] for x in self.target_pairs]
        self.lm_head = lm_head

        d_model = self.base_model.config.hidden_size
        dtype = next(self.base_model.parameters()).dtype

        self.lenses = nn.ModuleDict()

        for layer_idx in self.starts:
            
            lens = nn.Linear(d_model, d_model, dtype=dtype)
            nn.init.eye_(lens.weight)
            nn.init.zeros_(lens.bias)

            self.lenses[str(layer_idx)] = lens


    @property
    def _lenses(self):
        return self.lenses.module if hasattr(self.lenses, "module") else self.lenses


    @classmethod
    def from_checkpoint(cls, ckpt_path: str, start_layer: int, end_layer: int, base_model, lm_head, cfg=None, device="cuda"):
        """Load JumpQwen from a checkpoint with explicit start/end layers."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = cls(base_model, lm_head, cfg, target_pairs=[[start_layer, end_layer]])

        model.lenses.load_state_dict(sd)
        model.lenses.to(device)

        return model


    def forward(self, input_ids):

        embeds = self.base_model.embed_tokens(input_ids)
        seq_len = embeds.shape[1]
        pos_embs = compute_position_embeddings(self.base_model, embeds)

        causal_mask = make_causal_mask(seq_len, embeds.device, embeds.dtype)

        hiddens = embeds
        idx = 0

        while idx < len(self.layers):

            layer_mask = None if self.layer_types[idx] == "linear_attention" else causal_mask

            hiddens = self.base_model.layers[idx](
                hiddens,
                position_embeddings=pos_embs,
                attention_mask=layer_mask,
            )

            if idx in self.starts:
                hiddens = self._lenses[str(idx)](hiddens.detach())
                q = self.starts.index(idx)
                idx = self.ends[q]
            else:
                idx += 1

        jump_logits = self.lm_head(self.base_model.norm(hiddens))

        return jump_logits
