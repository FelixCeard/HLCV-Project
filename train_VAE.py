import logging
import os
import re

import matplotlib.pyplot as plt
import torch.optim
import wandb  # weight and bias for loss visualization etc...
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary

from model.custom_VAE import CustomVAE
from model.dataloader import ImageDataLoader

log_wandb = False

if log_wandb:
    wandb.init(project="test-project", entity="hlcv22")
logging.basicConfig(level=logging.INFO)

epochs = 10000
lr = 0.001
batch_size = 3
path_images = 'F:/DATASETS/original/images'
path_sketches = 'F:/DATASETS/original/sketches'
path_weights = './run/weights/'

transforms = []

def check_weight_path(path, folder=None):
    if os.path.isdir(path) == False:
        splited = re.split(r"/|\\", path)[:-1]
        folder_name = re.split(r"/|\\", path)[-1]
        check_weight_path('/'.join(splited), folder_name)
        if os.path.isdir(path) == False:
            os.mkdir(path)
    else:
        if folder is not None and os.path.isdir(os.path.join(path, folder)) == False:
            os.mkdir(os.path.join(path, folder))


check_weight_path(path_weights)


def make_new_folder(path_weights):
    if os.path.isdir(path_weights) == False:
        raise RuntimeError(f"The given path is not valid: {path_weights}")

    number_of_previous_runs = len(next(os.walk(path_weights))[1])
    new_path = os.path.join(path_weights, f'run-{number_of_previous_runs + 1}')

    os.mkdir(new_path)

    return new_path


path_weights = make_new_folder(path_weights)

# create the model
logging.info('initializing the model')
vae = CustomVAE(
    image_size=(360, 240),  # latent space: (45, 30)
    num_tokens=2048,
    codebook_dim=512,
    num_layers=3,
    num_resnet_blocks=3,
    hidden_dim=64, # to search 64(965,379), 128(3,517,955)
    channels=3,
    smooth_l1_loss=False,
    temperature=0.9,
    drop_out_rate=0.1
).to('cuda')
logging.info('done')

summary(vae, (3, 360, 240))

exit()


logging.info('loading the datasets')

dataset_image = ImageDataLoader(path_images=path_images)

train_percentage = 0.8  # 80% for training
train_size = int(len(dataset_image) * train_percentage)
test_size = len(dataset_image) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset_image, [train_size, test_size])

dataloader_train = DataLoader(
    train_dataset,
    batch_size=batch_size,  # batch size
    shuffle=True,
    num_workers=0,  # for multiprocessing
    collate_fn=None,
    pin_memory=False,
)
dataloader_test = DataLoader(
    test_dataset,
    batch_size=batch_size,  # batch size
    shuffle=False,  # no shuffle
    num_workers=0,  # for multiprocessing
    collate_fn=None,
    pin_memory=False,
)
logging.info('done')

# train

optimizer = torch.optim.Adagrad(params=vae.parameters(), lr=lr, weight_decay=0.1)


wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size
}

logging.info(f'Got {train_size} training images and {test_size} testing images')

# get 5 images for logging the display
n = batch_size

log_images_list = [test_dataset[i]['image'] for i in range(n)]
log_images = torch.zeros([len(log_images_list)] + list(log_images_list[0].shape))
for i in range(n):
    log_images[i] = log_images_list[i]
log_images = log_images.to('cuda')

best_loss_train = torch.tensor([99999999999999.0]).to('cuda')
best_loss_test = torch.tensor([99999999999999.0]).to('cuda')
logging.info('starting training')
for epoch in range(epochs):
    batch_i = 0

    # train
    logging.info('train...')
    epoch_loss = torch.zeros(1).to('cuda')
    if log_wandb:
        wandb.log({'epoch': epoch})

    for batch in tqdm(dataloader_train):

        if batch_i % 5 == 0:  # gradient accumulation:
            optimizer.zero_grad()

        imgs = batch['image'].to('cuda')

        loss = vae(imgs, return_loss=True)
        loss.backward()

        epoch_loss += loss

        if log_wandb:
            wandb.log({"step_loss": loss})

        optimizer.step()

        batch_i += 1

    epoch_loss /= train_size

    if epoch_loss < best_loss_train:
        best_loss_train = epoch_loss
        torch.save(vae, os.path.join(path_weights, f'best_train_model.pt'))

    logging.info('test...')
    # test
    with torch.no_grad():
        test_loss = torch.zeros(1).to('cuda')
        for batch in tqdm(dataloader_train):
            imgs = batch['image'].to('cuda')
            loss = vae(imgs, return_loss=True)
            test_loss += loss
        test_loss /= test_size

        if test_loss < best_loss_test:
            best_loss_test = test_loss
            torch.save(vae, os.path.join(path_weights, f'best_test_model.pt'))

    reconstructed = vae(log_images, return_loss=False)

    print(torch.min(reconstructed), torch.max(reconstructed))

    reconstructed = torch.clip(reconstructed, 0, 1)

    # print(reconstructed.shape) # [3, 3, 240, 360]

    # grid_tensor = torch.concat([log_images, reconstructed.resize(batch_size, 240, 360, 3)], 0) #, 0

    if log_wandb:
        fig, ax = plt.subplots(2, batch_size, figsize=(batch_size * 5, 10))
        for j in range(batch_size):
            ax[0, j].imshow(log_images[j].resize(240, 360, 3).detach().cpu())
            ax[1, j].imshow(reconstructed[j].resize(240, 360, 3).detach().cpu())

        log_info = {"loss": epoch_loss, 'epoch': epoch, 'test loss': test_loss, 'plot': plt}

        wandb.log(log_info)

    # save the weights after each epoch (given that it takes ~5 hours on my GPU -> ~2 hourse on RTX?)
    torch.save(vae, os.path.join(path_weights, f'weights_epoch_{epoch}.pt'))
