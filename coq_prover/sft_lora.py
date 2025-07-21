from coq_prover.coq_finetune.ft_lora import Trainer
import os

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()