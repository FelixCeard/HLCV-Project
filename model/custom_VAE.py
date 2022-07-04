#!/usr/bin/env python
from math import sqrt

import torch
import torch.nn.functional as F
from dalle_pytorch import distributed_utils
from dalle_pytorch.dalle_pytorch import eval_decorator, DiscreteVAE
from einops import rearrange
from matplotlib import pyplot as plt
from torch import nn, einsum

from model.dataloader import ImageDataLoader


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, mean=0.0, std=1e-3)
        m.bias.data.fill_(0.0)

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0.0)

class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class CustomVAE(DiscreteVAE):
    def __init__(
            self,
            image_size=(360, 240),  # (45, 30)
            num_tokens=2048,
            codebook_dim=512,
            num_layers=3,
            num_resnet_blocks=0,
            hidden_dim=64,
            channels=3,
            smooth_l1_loss=False,
            temperature=0.9,
            straight_through=False,
            kl_div_loss_weight=0.,
            normalization=((*((0.5,) * 3), 0), (*((0.5,) * 3), 1)),
            device='cuda',
            drop_out_rate=0
    ):
        super().__init__()
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.device = device

        self.channels = channels
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()))
            if drop_out_rate > 0:
                enc_layers.append(nn.Dropout(drop_out_rate))
                dec_layers.append(nn.Dropout(drop_out_rate))

            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            if drop_out_rate > 0:
                enc_layers.append(nn.Dropout(drop_out_rate))
                dec_layers.append(nn.Dropout(drop_out_rate))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = tuple(map(lambda t: t[:channels], normalization))

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (
                not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
            distributed_utils.DeepSpeedBackend)
        ):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.reshape(images.shape[0], 3, 240, 360).sub_(means).div_(stds)
        return images.reshape(images.shape[0], 3, 240, 360)

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def decode(
            self,
            img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(
            self,
            img,
            return_loss=False,
            return_recons=False,
            return_logits=False,
            temp=None
    ):
        # device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)

        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device=self.device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)

        loss = recon_loss + (kl_div * self.kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out




if __name__ == '__main__':
    vae = CustomVAE(
        image_size = (360, 240),
        num_layers = 3,           # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
        num_tokens = 8192,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
        codebook_dim = 512,       # codebook dimension
        hidden_dim = 64,          # hidden dimension
        num_resnet_blocks = 2,    # number of resnet blocks
        temperature = 0.9,        # gumbel softmax temperature, the lower this is, the harder the discretization
        straight_through = False, # straight-through for gumbel softmax. unclear if it is better one way or the other
    ).to('cuda')

    # test backwards pass
    path_images = 'F:/DATASETS/original/images'
    path_sketches = 'F:/DATASETS/original/sketches'

    dataset_image = ImageDataLoader(path_dataset=path_images)
    dataset_sketch = ImageDataLoader(path_dataset=path_sketches)

    # dataloader = DataLoader(
    #     dataset_image,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=None,
    #     pin_memory=False,
    #  )



    # train
    # for batch in dataloader:
    image = dataset_image[4]['image'].resize(1, 3, 360, 240).to('cuda')
    sketch = dataset_sketch[4]['image'].resize(1, 3, 360, 240).to('cuda')

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.resize(240, 360, 3).detach().cpu()/255)
    ax[1].imshow(sketch.resize(240, 360, 3).detach().cpu()/255)

    plt.show()
    #
    # # loss = vae(image, return_loss=True)
    print(vae.get_codebook_indices(image).shape)
    print(vae.get_codebook_indices(sketch).shape)
    #
    # print(dalle())
            # loss.backward()
            # break