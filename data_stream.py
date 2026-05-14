import os
from datasets import load_dataset, interleave_datasets
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, DataLoader
import torch
import time
import json


DATA_SOURCES = {
    "cc":       "train/RedPajamaCommonCrawl",
    "c4":       "train/RedPajamaC4",
    "github":   "train/RedPajamaGithub",
    "arxiv":    "train/RedPajamaArXiv",
    "wiki":     "train/RedPajamaWikipedia",
    "stackex":  "train/RedPajamaStackExchange",
}

DATA_PROBS = [0.40, 0.15, 0.10, 0.20, 0.10, 0.05]


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len: int,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        shuffle_buffer_size: int = 10_000,
        add_eos: bool = True,
        max_consecutive_fails: int = 20,
        fallback_log_dir: str = "streaming_fallbacks",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.add_eos = add_eos
        self.max_consecutive_fails = max_consecutive_fails

        os.makedirs(fallback_log_dir, exist_ok=True)
        self.fallback_log_file = os.path.join(fallback_log_dir, f"rank_{rank}.txt")

    def _log_fallback(self, payload: dict):
        payload = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rank": self.rank,
            "world_size": self.world_size,
            **payload,
        }

        with open(self.fallback_log_file, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _make_stream(self, seed_offset: int = 0):
        subsets = []
        base_seed = self.seed + seed_offset

        for j, data_dir in enumerate(DATA_SOURCES.values()):
            #ds = load_dataset(
            #    "MBZUAI-LLM/SlimPajama-627B-DC",
            #    data_dir=data_dir,
            #    split="train",
            #    streaming=True,
            #    trust_remote_code=True,
            #)

            ds = load_dataset(
                "json",
                data_files={
                    "train": f"hf://datasets/MBZUAI-LLM/SlimPajama-627B-DC/{data_dir}/*.jsonl.zst"
                },
                split="train",
                streaming=True,
)
            ds = ds.shuffle(
                seed=base_seed + j,
                buffer_size=self.shuffle_buffer_size,
            )

            ds = split_dataset_by_node(
                ds,
                rank=self.rank,
                world_size=self.world_size,
            )

            subsets.append(ds)

        return interleave_datasets(
            subsets,
            probabilities=DATA_PROBS,
            seed=base_seed,
            stopping_strategy="all_exhausted",
        )

    def __iter__(self):
        restart = 0
        seed_offset = 0

        stream = self._make_stream(seed_offset=seed_offset)
        it = iter(stream)

        buffer = []
        consecutive_fails = 0

        while True:
            try:
                sample = next(it)

            except StopIteration:
                return

            except Exception as e:
                consecutive_fails += 1

                if consecutive_fails < self.max_consecutive_fails:
                    wait = min(2 ** consecutive_fails, 120)

                    self._log_fallback({
                        "event": "retry_same_iterator",
                        "consecutive_fails": consecutive_fails,
                        "wait_seconds": wait,
                        "restart": restart,
                        "seed_offset": seed_offset,
                        "error": repr(e),
                    })

                    time.sleep(wait)
                    continue

                # Escape hatch:
                restart += 1
                seed_offset = 10_000 * restart

                self._log_fallback({
                    "event": "restart_stream_with_new_seed_offset",
                    "consecutive_fails": consecutive_fails,
                    "restart": restart,
                    "new_seed_offset": seed_offset,
                    "error": repr(e),
                    "note": (
                        "Strict non-overlap may be violated after this fallback; "
                        "some data may repeat or be skipped."
                    ),
                })

                stream = self._make_stream(seed_offset=seed_offset)
                it = iter(stream)

                buffer = []
                consecutive_fails = 0
                continue

            consecutive_fails = 0

            text = sample["text"]
            if not text:
                continue

            ids = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            if self.add_eos and self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)

            buffer.extend(ids)

            while len(buffer) >= self.seq_len:
                yield torch.tensor(
                    buffer[:self.seq_len],
                    dtype=torch.long,
                )
                buffer = buffer[self.seq_len:]


