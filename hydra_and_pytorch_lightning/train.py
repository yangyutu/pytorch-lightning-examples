import logging
import hydra

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    
    
    
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.train.seed)

    network = instantiate(cfg.network)
    data_module = instantiate(cfg.data)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model=network, datamodule=data_module)
    if cfg.train.run_test:
        trainer.test(datamodule=data_module)


# python -m leela_zero_pytorch.train +network=small
# override simple arguments
# python train.py trainer.max_epochs=4

# append a new argument for data batch size
# python train.py trainer.max_epochs=4 +data.batch_size=128

# change argument to allow run test
# python train.py trainer.max_epochs=4 +data.batch_size=128 network=complex train.run_test=true

# use a complex model instead of the simple one
# python train.py trainer.max_epochs=4 +data.batch_size=128 network=complex

# override project name in the first logger. 0 below is the first element in the logger list
# python train.py trainer.max_epochs=4 +data.batch_size=128 network=complex trainer.logger.0.project=new_tag

# override tags in the logger, need to use single quote since we have list here
# python train.py trainer.max_epochs=4 +data.batch_size=128 network=complex 'trainer.logger.0.tags=[hydra, new_tag]'

if __name__ == "__main__":
    main()
