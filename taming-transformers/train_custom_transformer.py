import datetime
import logging

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb

from utils import *

if __name__ == '__main__':
    download_pretrained()
    logging.debug('init')

    # wandb
    wandb.login(key='e0da967bc1f1f7b4895de7ecd6063d9513e0337c')
    wandb_logger = WandbLogger(project="[REAL]transformer", resume=True)

    # load configs
    logging.debug('loading configs')
    configs = [OmegaConf.load('./configs/custom_transformer.yaml')]
    config = OmegaConf.merge(*configs)

    # model
    logging.debug('loading model')
    model = instantiate_from_config(config.model)


    # data
    logging.debug('loading data')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()


    # dirs
    logging.debug('init callbacks')
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + 'custom_transformer'
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=ckptdir, filename="{epoch:06}", save_last=True),
        ImageLogger(batch_frequency=40000, max_images=7, clamp=True),
    ]

    # trainer
    accumulate_grad_batches = 2
    batch_size = config.data.params.batch_size
    model.learning_rate = accumulate_grad_batches * batch_size * config.model.base_learning_rate

    trainer = Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        auto_scale_batch_size=True,  # use it?
        auto_lr_find=True,            # use it?
        weights_save_path="",
        accumulate_grad_batches=accumulate_grad_batches,
        gpus=1
    )

    # train
    trainer.fit(model, data)

    # # test
    # trainer.test(model, data)

