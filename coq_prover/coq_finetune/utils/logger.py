import logging

import torch

# 创建一个 logger
logger = logging.getLogger(__name__)

# 创建一个自定义的 Formatter，包括日期前缀
formatter = logging.Formatter('%(levelname)s: %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 创建一个处理器（Handler），并将 Formatter 添加到处理器中
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# 将处理器添加到 logger
logger.addHandler(handler)
logger.setLevel(logging.INFO)  # 设置日志级别为 INFO

# 确保仅添加一次 handler
if not logger.hasHandlers():
    logger.addHandler(handler)


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        logger.info(msg)  # 使用 logger 输出信息
    elif is_rank_0():
        logger.info(msg)


def get_rank_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return -1
