from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from dalle_pytorch import distributed_utils
from dalle_pytorch.dalle_pytorch import eval_decorator, DiscreteVAE, set_requires_grad, always, SharedEmbedding, top_k, \
    gumbel_sample, prob_mask_like, is_empty
from dalle_pytorch.transformer import Transformer, DivideMax
from einops import rearrange
from matplotlib import pyplot as plt
from torch import nn, einsum

from model.dataloader import ImageDataLoader


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


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
            device='cuda'
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
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
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


class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        vae,
        num_text_tokens = 10000,
        text_seq_len = 256,
        depth,
        heads = 8,
        dim_head = 64,
        reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0,
        sparse_attn = False,
        attn_types = None,
        loss_img_weight = 7,
        stable = False,
        sandwich_norm = False,
        shift_tokens = True,
        rotary_emb = True,
        shared_attn_ids = None,
        shared_ff_ids = None,
        share_input_output_emb = False,
        optimize_for_inference = False,
    ):
        super().__init__()

        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_fmap_size = (vae.image_size // (2 ** vae.num_layers))
        image_seq_len = image_fmap_size ** 2

        num_text_tokens = num_text_tokens + text_seq_len  # reserve unique padding tokens for each position (text seq len)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) if not rotary_emb else always(0) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape = (image_fmap_size, image_fmap_size)) if not rotary_emb else always(0)

        self.num_text_tokens = num_text_tokens # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.vae = vae
        set_requires_grad(self.vae, False) # freeze VAE from being trained

        self.transformer = Transformer(
            dim = dim,
            causal = True,
            seq_len = seq_len,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            reversible = reversible,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            attn_types = attn_types,
            image_fmap_size = image_fmap_size,
            sparse_attn = sparse_attn,
            stable = stable,
            sandwich_norm = sandwich_norm,
            shift_tokens = shift_tokens,
            rotary_emb = rotary_emb,
            shared_attn_ids = shared_attn_ids,
            shared_ff_ids = shared_ff_ids,
            optimize_for_inference = optimize_for_inference,
        )

        self.stable = stable

        if stable:
            self.norm_by_max = DivideMax(dim = -1)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),
        )

        if share_input_output_emb:
            self.text_emb = SharedEmbedding(self.to_logits[1], 0, num_text_tokens)
            self.image_emb = SharedEmbedding(self.to_logits[1], num_text_tokens, total_tokens)
        else:
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            self.image_emb = nn.Embedding(num_image_tokens, dim)

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))
        )

        self.register_buffer('logits_mask', logits_mask, persistent=False)
        self.loss_img_weight = loss_img_weight


    @torch.no_grad()
    @eval_decorator
    def generate_texts(
        self,
        tokenizer,
        text = None,
        *,
        filter_thres = 0.5,
        temperature = 1.
    ):
        text_seq_len = self.text_seq_len
        if text is None or text == "":
            text_tokens = torch.tensor([[0]]).cuda()
        else:
            text_tokens = torch.tensor(tokenizer.tokenizer.encode(text)).cuda().unsqueeze(0)

        for _ in range(text_tokens.shape[1], text_seq_len):
            device = text_tokens.device

            tokens = self.text_emb(text_tokens)
            tokens += self.text_pos_emb(torch.arange(text_tokens.shape[1], device = device))

            seq_len = tokens.shape[1]

            output_transf = self.transformer(tokens)

            if self.stable:
                output_transf = self.norm_by_max(output_transf)

            logits = self.to_logits(output_transf)

            # mask logits to make sure text predicts text (except last token), and image predicts image

            logits_mask = self.logits_mask[:, :seq_len]
            max_neg_value = -torch.finfo(logits.dtype).max
            logits.masked_fill_(logits_mask, max_neg_value)
            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            text_tokens = torch.cat((text_tokens, sample[:, None]), dim=-1)

        padding_tokens = set(np.arange(self.text_seq_len) + (self.num_text_tokens - self.text_seq_len))
        texts = [tokenizer.tokenizer.decode(text_token, pad_tokens=padding_tokens) for text_token in text_tokens]
        return text_tokens, texts

    @torch.no_grad()
    @eval_decorator
    def generate_images(
        self,
        text,
        *,
        clip = None,
        filter_thres = 0.5,
        temperature = 1.,
        img = None,
        num_init_img_tokens = None,
        cond_scale = 1.,
        use_cache = False,
    ):
        vae, text_seq_len, image_seq_len, num_text_tokens = self.vae, self.text_seq_len, self.image_seq_len, self.num_text_tokens
        total_len = text_seq_len + image_seq_len

        text = text[:, :text_seq_len] # make sure text is within bounds
        out = text

        if exists(img):
            image_size = vae.image_size
            assert img.shape[1] == 3 and img.shape[2] == image_size and img.shape[3] == image_size, f'input image must have the correct image size {image_size}'

            indices = vae.get_codebook_indices(img)
            num_img_tokens = default(num_init_img_tokens, int(0.4375 * image_seq_len))  # OpenAI used 14 * 32 initial tokens to prime
            assert num_img_tokens < image_seq_len, 'number of initial image tokens for priming must be less than the total image token sequence length'

            indices = indices[:, :num_img_tokens]
            out = torch.cat((out, indices), dim = -1)

        prev_cache = None
        cache = {} if use_cache else None
        for cur_len in range(out.shape[1], total_len):
            is_image = cur_len >= text_seq_len

            text, image = out[:, :text_seq_len], out[:, text_seq_len:]

            logits = self.forward_with_cond_scale(text, image, cond_scale = cond_scale, cache = cache)
            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            sample -= (num_text_tokens if is_image else 0) # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            out = torch.cat((out, sample[:, None]), dim=-1)

        text_seq = out[:, :text_seq_len]

        img_seq = out[:, -image_seq_len:]
        images = vae.decode(img_seq)

        if exists(clip):
            scores = clip(text_seq, images, return_loss = False)
            return images, scores

        return images

    def forward_with_cond_scale(self, *args, cond_scale = 1, cache = None, **kwargs):
        if cond_scale == 1:
            return self(*args, **kwargs)

        prev_cache = cache.copy() if exists(cache) else None
        logits = self(*args, cache = cache, **kwargs)

        # discovery by Katherine Crowson
        # https://twitter.com/RiversHaveWings/status/1478093658716966912
        null_cond_logits = self(*args, null_cond_prob = 1., cache = prev_cache, **kwargs)
        return null_cond_logits + (logits - null_cond_logits) * cond_scale

    def forward(
        self,
        text,
        image = None,
        return_loss = False,
        null_cond_prob = 0.,
        cache = None,
    ):
        assert text.shape[-1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
        batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len

        # randomly remove text condition with <null_cond_prob> probability

        if null_cond_prob > 0:
            null_mask = prob_mask_like((batch,), null_cond_prob, device = device)
            text *= rearrange(~null_mask, 'b -> b 1')

        # make sure padding in text tokens get unique padding token id

        text_range = torch.arange(self.text_seq_len, device = device) + (self.num_text_tokens - self.text_seq_len)
        text = torch.where(text == 0, text_range, text)

        # add <bos>

        text = F.pad(text, (1, 0), value = 0)

        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device = device))

        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4

            if is_raw_image:
                image_size = self.vae.image_size
                channels = self.vae.channels
                assert tuple(image.shape[1:]) == (channels, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'

                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_emb(image)

            image_emb += self.image_pos_emb(image_emb)

            tokens = torch.cat((tokens, image_emb), dim = 1)

            seq_len += image_len

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained

        if tokens.shape[1] > total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

        if self.stable:
            alpha = 0.1
            tokens = tokens * alpha + tokens.detach() * (1 - alpha)

        if exists(cache) and cache.get('offset'):
            tokens = tokens[:, -1:]
        out = self.transformer(tokens, cache=cache)

        if self.stable:
            out = self.norm_by_max(out)

        logits = self.to_logits(out)

        # mask logits to make sure text predicts text (except last token), and image predicts image

        logits_mask = self.logits_mask[:, :seq_len]
        if exists(cache) and cache.get('offset'):
            logits_mask = logits_mask[:, -1:]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if exists(cache):
            cache['offset'] = cache.get('offset', 0) + logits.shape[1]

        if not return_loss:
            return logits

        assert exists(image), 'when training, image must be supplied'

        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim = 1)

        logits = rearrange(logits, 'b n c -> b c n')

        loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
        loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])

        loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
        return loss




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

    dalle = DALLE(
        dim=1350,
        vae=vae,  # automatically infer (1) image sequence length and (2) number of image tokens
        num_text_tokens=8192,  # vocab size for text
        text_seq_len=1350,  # text sequence length
        depth=12,  # should aim to be 64
        heads=16,  # attention heads
        dim_head=64,  # attention head dimension
        attn_dropout=0.1,  # attention dropout
        ff_dropout=0.1  # feedforward dropout
    )

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