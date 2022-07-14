import argparse
import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader
import requests
import shutil


from taming.data.utils import custom_collate


def download_pretrained():
    # download the f16 model
    os.makedirs('./logs/vqgan_imagenet_f16_1024/checkpoints', exist_ok=True)
    os.makedirs('./logs/vqgan_imagenet_f16_1024/configs', exist_ok=True)

    url1 = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
    url2 = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'
    response = requests.get(url1, stream=True)
    with open('./logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

    response = requests.get(url2, stream=True)
    with open('./logs/vqgan_imagenet_f16_1024/configs/model.yaml', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response




class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)




class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, logger=None):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

        self._logger = logger

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        # raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            if images[k].shape[1] != 3:
                images[k] = images[k].permute(0, 2, 1, 3)
            grid = torchvision.utils.make_grid(images[k])
            fig, ax = plt.subplots(1, images[k].shape[0])
            for i in range(images[k].shape[0]):
                ax[i].imshow(torch.clip(images[k][i].permute(1, 2, 0), 0, 1))
            # grid = grid.detach().cpu()  # .numpy()
            # print(grid.shape)
            # t = T.ToPILImage()
            # img = t(grid)

            # img = Image.fromarray(grid)
            # print(img.size)

            # img.save(f'./img{k}.png')

            # pl.loggers.wandb.wandb.log({f'{split}/{k}': wandb.Image(img)})
            # pl_module.logger.experiment.log({'image': wandb.Image(grid)})
            pl_module.logger.experiment.log({f'{split}/{k}': plt})

            fig.clear()
            plt.close(fig)
            # pl_module.logger.experiment.log({"val_input_image": [wandb.Image(img, caption=k)]})
            # self._logger.log_image("val_input_image", images=[wandb.Image(grid)])
            # wandb.log({"val_input_image": [wandb.Image(img, caption=k)]})

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            # print('images1:', images[k].shape)
            if images[k].shape[1] != 3:
                images[k] = images[k].permute(0, 2, 1, 3)  # (2, 224, 3, 224) -> (2, 3, 224, 224)
            # print('images2:', images[k].shape)
            # grid = torchvision.utils.make_grid(images[k])
            grid = (images[k] + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            # print('grid:', grid.shape)

            tag = f"{split}/{k}"

            for i in range(grid.shape[0]):
                pl_module.logger.experiment.add_image(
                    tag, grid[i],
                    global_step=pl_module.global_step)

            # print('done')

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        pass
        # root = os.path.join(save_dir, "images", split)
        # # print('images type', type(images))
        # for k in images:
        #     # print('images shape:', images[k].shape)
        #     if images[k].shape[1] != 3:
        #         images[k] = images[k].permute(0, 2, 1, 3)  # from (2, 224, 3, 224) to (2, 3, 224, 224)
        #     grid = torchvision.utils.make_grid(images[k], nrow=4)
        #
        #     grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        #     grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        #     grid = grid.numpy()
        #     grid = (grid * 255).astype(np.uint8)
        #     filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
        #         k,
        #         global_step,
        #         current_epoch,
        #         batch_idx)
        #     path = os.path.join(root, filename)
        #     os.makedirs(os.path.split(path)[0], exist_ok=True)
        #     # print('grid:', grid.shape)
        #     Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            # logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            self._wandb(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print('TRAIN BATCH END')
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))