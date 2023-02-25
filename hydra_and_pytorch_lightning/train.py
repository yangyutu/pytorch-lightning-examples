import logging
import hydra

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Trainer:
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    network = instantiate(cfg.network)
    data_module = instantiate(cfg.data)
    # trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = instantiate(cfg.trainer)
    # trainer = Trainer(**cfg.pl_trainer, logger=trainer_logger)
    # callbacks = [cfg.pl_trainer]
    trainer.fit(model=network, datamodule=data_module)
    # if cfg.train.run_test:
    #     trainer.test(datamodule=data)

    # return trainer


# python -m leela_zero_pytorch.train +network=small
# override simple arguments
# python train.py trainer.max_epochs=4

# append a new argument for data batch size
# python train.py trainer.max_epochs=4 +data.batch_size=128

# use a complex model instead of the simple one
# python train.py trainer.max_epochs=4 +data.batch_size=128 network=complex

if __name__ == "__main__":
    main()
