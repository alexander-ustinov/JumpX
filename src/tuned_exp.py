import os
from omegaconf import OmegaConf as Om
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import JumpQwen
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from data_stream import StreamingDataset
from torch.nn.parallel import DistributedDataParallel as DDP


class JumpExperiment():
    
    def __init__(self, config: str):
        
        self.cfg = Om.load(config)
        Om.resolve(self.cfg)

        self._init_distributed()

        if self.rank == 0:
            self._download_model()
        dist.barrier()

        self.base_model, self.tokenizer, self.lm_head = self.prepare_model_and_tokenizer()
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
        """
        llm = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.name,
            dtype=getattr(torch, self.cfg.model.dtype),
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(self.device).eval()

        text_model = llm.model
        lm_head = llm.lm_head

        for p in llm.parameters():
            p.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name, trust_remote_code=True)

        return text_model, tokenizer, lm_head


    def _init_distributed(self):
        
        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.device = torch.device(f"cuda:{self.rank}")
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
