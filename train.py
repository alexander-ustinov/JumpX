import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire
from trainer import JumpTrainer


def main(config: str):
    trainer = JumpTrainer(config)
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
