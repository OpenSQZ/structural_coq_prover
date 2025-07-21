from coq_prover.coq_finetune.ft_full import Trainer
import os

# os.environ["HTTP_PROXY"] = ""
# os.environ["HTTPS_PROXY"] = ""

my_env = os.environ.copy()
my_env["PATH"] = "/root/miniconda3/envs/coq/bin:" + my_env["PATH"]
os.environ.update(my_env)

import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()