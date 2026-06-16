import os
from omegaconf import OmegaConf as Om
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import JumpQwen
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from data_stream import StreamingDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh


class JumpExperiment():
    
    def __init__(self, config: str):
        
        self.cfg = Om.load(config)
        Om.resolve(self.cfg)

        self._init_distributed()

        if self.rank == 0:
            self._download_model()
        dist.barrier()

        self.base_model, self.tokenizer, self.lm_head = self.prepare_model_and_tokenizer()
        self._shard_base_model()
        dist.barrier()

        self.JumpQwen = JumpQwen(self.base_model, self.lm_head, self.cfg)

        self.JumpQwen.lenses = self.JumpQwen.lenses.to(self.device)
        self.JumpQwen.lenses = DDP(self.JumpQwen.lenses, device_ids=[self.rank])
        self.lenses = self.JumpQwen.lenses

        self.muon_params, self.adam_params = [], []
        for element in self.lenses.parameters():
            (self.muon_params if element.ndim >= 2 else self.adam_params).append(element)
   
        self.trainable_params = self.muon_params + self.adam_params

        self.dataloader = self.create_loader()


    def _download_model(self):
        """Rank 0 downloads/caches weights before other ranks try to load."""
        AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            torch_dtype=getattr(torch, self.cfg.model.dtype),
            trust_remote_code=True,
        )
        AutoTokenizer.from_pretrained(self.cfg.model.name, trust_remote_code=True)

    def prepare_model_and_tokenizer(self):

        """
        Load Qwen3.5-type model and returns text_model.

        The model is loaded on CPU (no .to(device)); _shard_base_model then
        materialises only each rank's shard on the GPU, so the full ~18GB model
        never lands on device at once. from_pretrained loads on CPU so dtype and
        all (incl. non-persistent) buffers are set correctly before sharding.
        """
        llm = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            dtype=getattr(torch, self.cfg.model.dtype),
            attn_implementation=self.cfg.model.get("attn_implementation", "sdpa"),
            trust_remote_code=True,
        ).eval()

        text_model = llm.model
        lm_head = llm.lm_head

        for p in llm.parameters():
            p.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name, trust_remote_code=True)

        return text_model, tokenizer, lm_head


    def _shard_base_model(self):
        """FSDP2-shard the frozen transformer layers across ranks.

        Only the decoder layers (the bulk of the params) are sharded; they are
        called both by the teacher's full forward and by JumpQwen's manual
        per-layer forward, and per-module FSDP hooks fire in both cases.
        The base model is loaded on CPU; fully_shard places each rank's shard
        directly on the GPU, so the full model is never materialised on device.
        embed_tokens / norm / lm_head stay replicated (small, frozen), and the
        trainable lenses are left on DDP so Muon sees full (unsharded) matrices.
        """
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated() / 1e9

        if self.world_size >= 2:
            mesh = init_device_mesh("cuda", (self.world_size,))
            for layer in self.base_model.layers:
                # reshard_after_forward=True is essential here: each layer is its
                # own FSDP root (no parent FSDP module wraps them), and roots
                # otherwise default to keeping params gathered after forward —
                # which would re-materialise the whole model and wipe out the
                # sharding. fully_shard moves the local shard to the mesh (GPU).
                fully_shard(layer, mesh=mesh, reshard_after_forward=True)

        # Move the still-on-CPU replicated params/buffers (embed_tokens, norm,
        # rotary buffers, lm_head) to the GPU. Sharded layer params are already
        # on the mesh device, so .to() is a no-op for them.
        self.base_model.to(self.device)
        self.lm_head.to(self.device)

        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated() / 1e9
        if self.is_main_process:
            print(f"[FSDP] num_layers={len(self.base_model.layers)} "
                  f"gpu_allocated before_shard={before:.2f}GB after_shard={after:.2f}GB",
                  flush=True)

    def _init_distributed(self):
        
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)
        self.world_size = dist.get_world_size()
        self.is_main_process = self.rank == 0

    def create_loader(self):

        ds = StreamingDataset(self.tokenizer, self.cfg.trainer.seq_len, rank=self.rank, world_size=self.world_size)

        loader = DataLoader(
            ds,
            batch_size=self.cfg.trainer.batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

        return loader
