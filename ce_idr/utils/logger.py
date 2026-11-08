import logging
import os
from torch.utils.tensorboard import SummaryWriter

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    writer = SummaryWriter(log_dir)
    return (logger, writer)
