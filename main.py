import numpy as np
import tensorflow as tf

from trainer import Trainer
from config import get_config
from data_loader import get_loader
# from data_loader import get_fashion_loader
from utils import prepare_dirs_and_logger, save_config


def main(config):
    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)
    loader = get_loader(config.data_dir, config.batch_size)
    trainer = Trainer(config, loader)
    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        trainer = Trainer(config, loader)
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()


if __name__ == "__main__":
    print("main")
    config, unparsed = get_config()
    main(config)
