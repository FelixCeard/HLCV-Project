import torch.optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from model.custom_VAE import CustomVAE
from model.dataloader import ImageSketchDataLoader, ImageDataLoader

import logging
import wandb  # weight and bias for loss visualization etc...

log_wandb = True

if log_wandb:
    wandb.init(project="test-project", entity="hlcv22")
logging.basicConfig(level=logging.INFO)

epochs = 10000
lr = 0.001
batch_size = 3

# create the model
logging.info('initializing the model')
vae = CustomVAE(
    image_size=(360, 240),  # latent space: (45, 30)
    num_tokens=2048,
    codebook_dim=512,
    num_layers=3,
    num_resnet_blocks=0,
    hidden_dim=64,
    channels=3,
    smooth_l1_loss=False,
    temperature=0.9,
).to('cuda')
logging.info('done')

logging.info('loading the datasets')
path_images = 'F:/DATASETS/original/images'
path_sketches = 'F:/DATASETS/original/sketches'

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

optimizer = torch.optim.Adagrad(params=vae.parameters(), lr=lr)

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


logging.info('starting training')
for epoch in range(epochs):
    batch_i = 0

    # train
    logging.info('train...')
    epoch_loss = torch.zeros(1).to('cuda')
    if log_wandb:
        wandb.log({'epoch': epoch})
    for batch in dataloader_train:

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

    logging.info('test...')
    # test
    with torch.no_grad():
        test_loss = torch.zeros(1).to('cuda')
        for batch in dataloader_train:
            imgs = batch['image'].to('cuda')
            loss = vae(imgs, return_loss=True)
            test_loss += loss
        test_loss /= test_size
    if log_wandb:
        wandb.log({})

    reconstructed = vae(log_images, return_loss=False)

    grid_tensor = torch.concat([log_images, reconstructed.resize(batch_size, 240, 360, 3)], 0) #, 0
    # print(grid_tensor.shape)
    # grid = make_grid(grid_tensor, nrow=batch_size)

    if log_wandb:
        grid_list = {}
        for i in range(batch_size*2):
            wandb_img = wandb.Image(grid_tensor[i].resize(3, 240, 360), caption="Top: Output, Bottom: Input")
            grid_list[f'image_{i}'] = wandb_img
        logging.info('print image')

        log_info = {"loss": epoch_loss, 'epoch': epoch, 'test loss': test_loss}
        for key in grid_list.keys():
            log_info[key] = grid_list[key]

        wandb.log(log_info)



