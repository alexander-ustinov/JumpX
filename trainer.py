import os
import json
import contextlib
from collections import defaultdict, deque
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW, Muon
from src.tuned_exp import JumpExperiment
from tqdm import tqdm
from X.eval.quality_loader import load_quality
from eval.eval_utils import make_quality_prompt, prepare_mmlu


class JumpTrainer:

    def __init__(self, config: str):

        self.experiment = JumpExperiment(config)
        self.cfg = self.experiment.cfg

        self.opt_muon = Muon(self.experiment.muon_params, lr=self.cfg.trainer.lr_muon, momentum=0.95, weight_decay=0.01)
        self.opt_adam = AdamW(self.experiment.adam_params, lr=self.cfg.trainer.lr_adam, betas=(0.9, 0.95), weight_decay=0.01)
        self.prev_window = deque(maxlen=self.cfg.trainer.log_prev_steps)

        self.eval_mmlu_data = prepare_mmlu(self.experiment.tokenizer, self.cfg.eval_mmlu.N)
        self.LABEL_LIST = ["A", "B", "C", "D"]

        self.letter_ids = [
            self.experiment.tokenizer.encode(" " + letter, add_special_tokens=False)[0]
            for letter in self.LABEL_LIST 
        ]

    def train_step(self, batch, sync_gradients: bool):

        ctx = contextlib.nullcontext() if sync_gradients else self.experiment.lenses.no_sync()

        with ctx:
            batch = batch.to(self.experiment.device)

            teacher_logits = self.compute_teacher_logits(batch)
            V = teacher_logits.shape[-1]
            teacher_probs = F.softmax(teacher_logits.reshape(-1, V).float(), dim=-1)
            del teacher_logits

            jump_logits = self.experiment.JumpQwen(batch)

            loss = F.kl_div(
                F.log_softmax(jump_logits.reshape(-1, V).float(), dim=-1),
                teacher_probs,
                reduction="batchmean",
            ) / self.cfg.trainer.accum_steps

            loss.backward()

        return loss.item()

    def train(self):

        self.experiment.lenses.train()
        is_main = self.experiment.is_main_process

        pbar = tqdm(total=self.cfg.trainer.train_steps, desc="Training", disable=not is_main)
        data_iter = iter(self.experiment.dataloader)
        effective_step = 0

        while effective_step < self.cfg.trainer.train_steps:

            self.opt_muon.zero_grad()
            self.opt_adam.zero_grad()

            step_loss = 0.0

            for accum_idx in range(self.cfg.trainer.accum_steps):
                batch = next(data_iter)
                sync_gradients = (accum_idx == self.cfg.trainer.accum_steps - 1)
                micro_loss = self.train_step(batch, sync_gradients)
                step_loss += micro_loss

            grad_norm = self.compute_grad_norm()
            torch.nn.utils.clip_grad_norm_(self.experiment.trainable_params, 1.0)

            self.opt_muon.step()
            self.opt_adam.step()

            
            self.prev_window.append(step_loss)

            if effective_step % self.cfg.trainer.log_steps == 0:
                avg_loss = sum(self.prev_window) / len(self.prev_window)
                self.log(effective_step, step_loss, avg_loss, grad_norm)

            if effective_step > 0 and  effective_step % self.cfg.trainer.save_steps == 0:
                self.save_checkpoint(effective_step)

            if effective_step % self.cfg.trainer.eval_steps == 0:
                mmlu_metrics = self.eval_mmlu(effective_step)
                quality_metrics = self.eval_quality(effective_step)
                self.log_eval(effective_step, mmlu_metrics, quality_metrics)
                self.experiment.lenses.train()
                torch.cuda.empty_cache()

            effective_step += 1
            pbar.update(1)

    @torch.no_grad()
    def compute_teacher_logits(self, batch):
       
        teacher_out = self.experiment.base_model(input_ids=batch, use_cache=False).last_hidden_state
        teacher_logits = self.experiment.lm_head(teacher_out)

        return teacher_logits

    def log(self, step, step_loss, avg_loss, grad_norm):

        if not self.experiment.is_main_process:
            return

        parts = [f"step={step}"]
        parts.append(f"step_loss={step_loss:.6f}")
        parts.append(f"avg_loss={avg_loss:.6f}")
        parts.append(f"grad_norm={grad_norm:.4f}")

        line = " | ".join(parts)
        with open(self.cfg.trainer.log_file, "a") as f:
            f.write(line + "\n")


    def compute_grad_norm(self):

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in self.experiment.lenses.parameters()
            if p.grad is not None
        ) ** 0.5

        return grad_norm


    @torch.no_grad()
    def eval_mmlu(self, step: int):
       
        self.experiment.lenses.eval()

        rank = self.experiment.rank
        world_size = self.experiment.world_size
        shard = self.eval_mmlu_data[rank::world_size]

        local_correct = 0

        for example in tqdm(shard, desc="MMLU eval", disable=rank != 0):

            input_ids = example["input_ids"].unsqueeze(0).to(self.experiment.device)
            label_letter = example["label_letter"]

            logits = self.experiment.JumpQwen(input_ids)
            candidate_logits = logits[0, -1, self.letter_ids].float()
            pred_idx = candidate_logits.argmax().item()
            pred_letter = self.LABEL_LIST[pred_idx]

            if pred_letter == label_letter:
                local_correct += 1

        correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=self.experiment.device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

        return {"accuracy": correct_tensor.item() / len(self.eval_mmlu_data)}


    @torch.no_grad()
    def eval_quality(self, step: int): 
        
        n = self.cfg.eval_quality.N
        ds = load_quality(html_stripped=True)
        dataset = ds["dev"].select(range(min(n, len(ds["dev"]))))

        self.experiment.lenses.eval()

        rank = self.experiment.rank
        world_size = self.experiment.world_size
        indices = list(range(rank, n, world_size))

        local_correct = 0
        local_total = 0

        for i in tqdm(indices, desc="QuALITY eval", disable=rank != 0):
            ex = dataset[i]

            if ex["label"] is None:
                continue

            prompt = make_quality_prompt(ex)

            input_ids = self.experiment.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]

            if input_ids.shape[1] > self.cfg.eval_quality.max_len:
                continue

            input_ids = input_ids.to(self.experiment.device)
            logits = self.experiment.JumpQwen(input_ids)
            answer_logits = logits[0, -1, self.letter_ids]
            pred = int(answer_logits.argmax().item())

            gold = int(ex["label"])
            local_correct += int(pred == gold)
            local_total += 1

        correct_tensor = torch.tensor([local_correct], dtype=torch.long, device=self.experiment.device)
        total_tensor = torch.tensor([local_total], dtype=torch.long, device=self.experiment.device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        total = total_tensor.item()
        return {"accuracy": correct_tensor.item() / total if total > 0 else 0.0}


    def log_eval(self, step: int, metrics_mmlu: dict, metrics_quality: dict):

        if not self.experiment.is_main_process:
            return
        
        with open(self.cfg.trainer.eval_log_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "mmlu_accuracy": metrics_mmlu["accuracy"],
                        "quality_accuracy": metrics_quality["accuracy"],
                    }
                )
                + "\n"
            )
    

    def save_checkpoint(self, step):

        if not self.experiment.is_main_process:
            return

        os.makedirs(self.cfg.trainer.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.trainer.checkpoint_dir, f"step_{step}.pt")
        torch.save(self.experiment.lenses.module.state_dict(), path)
