"""
VAE training: input is sketch, output is image
"""

import logging
import os
import re

import matplotlib.pyplot as plt
import torch.optim
import wandb  # weight and bias for loss visualization etc...
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary

from model.custom_VAE import CustomVAE, weights_init
from model.dataloader import ImageDataLoader, ImageSketchDataLoader
import pytorch_warmup as warmup

logging.basicConfig(level=logging.INFO)

sweep_config = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'loss'},
    'parameters': {
        'batch_size': {'value': 3},  # find best batch size
        'epochs': {'value': 15},
        'learning_rate': {'distribution': 'uniform',
                          'max': 0.01,
                          'min': 0.00005},
        'optimizer': {'values': ['adam', 'sgd', 'adagrad']},
        'num_resnet_blocks': {'values': [1, 2, 3, 5, 7]},  # probably going to find 1 or 3
        'num_tokens': {'values': [512, 1024, 2048, 4086, 8192]},
        'num_gradient_accumulations': {'values': [1, 3, 5, 10, 15]},
        'hidden_dim': {'values': [32, 64, 128]},
        'drop_out': {'values': [0, 0.1, 0.2, 0.3]}
    }
}


def check_weight_path(path, folder=None):
    if not os.path.isdir(path):
        splited = re.split(r"[/\\]", path)[:-1]
        folder_name = re.split(r"[/\\]", path)[-1]
        check_weight_path('/'.join(splited), folder_name)
        if os.path.isdir(path) == False:
            os.mkdir(path)
    else:
        if folder is not None and os.path.isdir(os.path.join(path, folder)) == False:
            os.mkdir(os.path.join(path, folder))


def make_new_folder(path_weights):
    if not os.path.isdir(path_weights):
        raise RuntimeError(f"The given path is not valid: {path_weights}")

    number_of_previous_runs = len(next(os.walk(path_weights))[1])
    new_path = os.path.join(path_weights, f'run-{number_of_previous_runs + 1}')

    os.mkdir(new_path)

    return new_path


def train(config=None,
          path_images='./images',
          path_sketches='./sketches',
          path_weights='./run/weights/'):
    # check_weight_path(path_weights)
    # path_weights = make_new_folder(path_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using devide: {device}')

    with wandb.init(project="hyperparameter_search", entity="hlcv22", config=config):
        config = wandb.config
        # create the model
        logging.info('initializing the model')
        vae = CustomVAE(
            image_size=(360, 240),
            num_tokens=config.num_tokens,
            codebook_dim=512,
            num_layers=3,  # latent space: (45, 30)
            num_resnet_blocks=config.num_resnet_blocks,
            hidden_dim=config.hidden_dim,  # to search 64(965,379), 128(3,517,955)
            channels=3,
            smooth_l1_loss=False,
            temperature=0.9,
            drop_out_rate=config.drop_out
        ).to(device)

        vae.apply(weights_init)

        logging.info('done')
        # wandb.watch(vae, vae.loss_fn, log="all")

        # display a summary for information
        summary(vae, (3, 360, 240))

        logging.info('loading the datasets')
        dataset_image = ImageSketchDataLoader(path_images=path_images, path_sketches=path_sketches)

        train_percentage = 0.8  # 80% for training
        train_size = int(len(dataset_image) * train_percentage)
        test_size = len(dataset_image) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset_image, [train_size, test_size])

        dataloader_train = DataLoader(
            train_dataset,
            batch_size=config.batch_size,  # batch size
            shuffle=True,
            num_workers=0,  # for multiprocessing
            collate_fn=None,
            pin_memory=False,
        )
        dataloader_test = DataLoader(
            test_dataset,
            batch_size=config.batch_size,  # batch size
            shuffle=False,  # no shuffle
            num_workers=0,  # for multiprocessing
            collate_fn=None,
            pin_memory=False,
        )
        logging.info('done')
        logging.info(f'Got {train_size} training images and {test_size} testing images')

        num_steps = len(dataloader_train) * config.epochs

        # get the right optimizer
        logging.info('chosing optimizer')
        optimizer = torch.optim.Adagrad(params=vae.parameters(), lr=config.learning_rate)
        if config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params=vae.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'adam':
            optimizer = torch.optim.Adam(params=vae.parameters(), lr=config.learning_rate)

        # add a warmup and a decay for the learning rate
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.9,
                                                         total_iters=num_steps)
        warmup_scheduler = warmup.LinearWarmup(optimizer, 400)  # increase to 2000 for the long run
        logging.info(f'chosen optimizer: {config.optimizer}')
        logging.info('done')

        # saving images for the logging
        logging.info('loading logging images')
        n = config.batch_size
        log_images_list = [test_dataset[i]['image'] for i in range(n)]
        log_sketches_list = [test_dataset[i]['sketch'] for i in range(n)]

        log_images = torch.zeros([len(log_images_list)] + list(log_images_list[0].shape))
        for i in range(n):
            log_images[i] = log_images_list[i]
        log_images = log_images.to(device)

        log_sketch = torch.zeros([len(log_sketches_list)] + list(log_sketches_list[0].shape))
        for i in range(n):
            log_sketch[i] = log_sketches_list[i]
        log_sketch = log_sketch.to(device)
        logging.info('loading logging images')

        # for the loss logging
        best_loss_train = torch.tensor([99999999999999.0]).to(device)
        best_loss_test = torch.tensor([99999999999999.0]).to(device)
        logging.info('starting training')

        # training loop
        for epoch in range(1, config.epochs + 1):

            batch_i = 0

            logging.info('train...')
            epoch_loss = torch.zeros(1).to(device)
            wandb.log({'epoch': epoch, 'num_tokens': config.num_tokens, 'num_resnet_blocks': config.num_resnet_blocks})

            for batch in tqdm(dataloader_train):

                if batch_i % config.num_gradient_accumulations == 0:  # gradient accumulation:
                    optimizer.zero_grad()

                # get the images
                sketch = batch['sketch'].to(device)
                img = batch['image'].to(device)

                # predict
                predictions = vae(img, return_loss=False)

                # compute the raw difference
                loss = vae.loss_fn(predictions.permute(0, 2, 3, 1), img)
                loss.backward()

                # add the loss for logging
                epoch_loss += loss

                # perform the optimization
                optimizer.step()
                with warmup_scheduler.dampening():
                    lr_scheduler.step()

                wandb.log({"step loss": loss, 'learning rate': lr_scheduler.get_lr()[0]})

                batch_i += 1

            # mean epoch loss
            epoch_loss /= train_size

            # early stopping mechanism
            if epoch_loss < best_loss_train:
                best_loss_train = epoch_loss
                # torch.save(vae, os.path.join(path_weights, f'best_train_model.pt'))

            # test
            logging.info('test...')
            with torch.no_grad():

                test_loss = torch.zeros(1).to(device)

                for batch in tqdm(dataloader_test):
                    img = batch['image'].to(device)
                    sketch = batch['sketch'].to(device)
                    pred = vae(img, return_loss=False)

                    # test loss
                    test_loss += vae.loss_fn(pred.permute(0, 2, 3, 1), img)

                # mean test loss
                test_loss /= test_size

                # early stopping
                if test_loss < best_loss_test:
                    best_loss_test = test_loss
                    # torch.save(vae, os.path.join(path_weights, f'best_test_model.pt'))

            # prediction logging
            reconstructed = vae(log_images, return_loss=False)
            logging.debug(f'predicted values in the range [{torch.min(reconstructed)}, {torch.max(reconstructed)}]')
            reconstructed = torch.clip(reconstructed, 0, 1)

            fig, ax = plt.subplots(2, config.batch_size, figsize=(config.batch_size * 5, 10))
            # ax[0, 0].set_xlabel('Sketch')
            ax[0, 0].set_xlabel('Real image')
            ax[1, 0].set_xlabel('Predicted image')
            reconstructed = reconstructed.permute(0, 2, 3, 1)
            for j in range(config.batch_size):
                # ax[0, j].imshow(log_sketch[j].permute(1, 2, 0).detach().cpu())
                ax[0, j].imshow(log_images[j].detach().cpu())
                ax[1, j].imshow(reconstructed[j].detach().cpu())

            # stats logging
            log_info = {"loss": epoch_loss, 'epoch': epoch, 'test loss': test_loss, 'plot': plt}
            wandb.log(log_info)

            fig.clear()
            plt.close(fig)

            # save the weights after each epoch (given that it takes ~5 hours on my GPU -> ~2 hourse on RTX?)
            # torch.save(vae, os.path.join(path_weights, f'weights_epoch_{epoch}.pt'))


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="hyperparameter_search")
    wandb.agent(sweep_id, function=train, count=5)
