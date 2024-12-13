# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 10:32 am
# @Author  : Xiangzhu Lv
# @FileName: prompts5.py
# @Software: PyCharm

from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import matplotlib
matplotlib.use('TkAgg')


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


writer = SummaryWriter("logs")


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, ctx=False):
        if ctx:
            # prompts = prompts.unsqueeze(0).expand(1, -1, -1)
            x = prompts + self.positional_embedding.type(self.dtype)[1:25,:] # 9, 17, 25
        else:
            x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if not ctx:
            x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if ctx_init:
        #     # use given words to initialize context vectors
        #     ctx_init = ctx_init.replace("_", " ")
        #     n_ctx = len(ctx_init.split(" "))
        #     prompt = clip.tokenize(ctx_init)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(dtype)
        #     ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        #     prompt_prefix = ctx_init
        if ctx_init:
            ctx_init = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            ctx_init = ctx_init.replace(" {}.", "")
            ctx_init = ctx_init.replace("_", " ")
            prompt_n_ctx = len(ctx_init.split(" "))

            assert n_ctx >= prompt_n_ctx, f"#tokens ({n_ctx}) should larger equal than #initial prompt tokens ({prompt_n_ctx}, {ctx_init})"

            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = torch.zeros(n_ctx, ctx_dim, dtype=dtype)

            ctx_vectors[n_ctx - prompt_n_ctx:, :] = embedding[0, 1:1 +
                                                              prompt_n_ctx, :]
            prompt_prefix = " ".join(["X"] * (n_ctx - prompt_n_ctx))
            prompt_prefix = f"{prompt_prefix} {ctx_init}"
        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors1 = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors2 = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors3 = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors4 = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors5 = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors1 = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors2 = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors3 = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors4 = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors5 = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors1, std=0.02)
            nn.init.normal_(ctx_vectors2, std=0.02)
            nn.init.normal_(ctx_vectors3, std=0.02)
            nn.init.normal_(ctx_vectors4, std=0.02)
            nn.init.normal_(ctx_vectors5, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx1 = nn.Parameter(ctx_vectors1)  # to be optimized
        self.ctx2 = nn.Parameter(ctx_vectors2)  # to be optimized
        self.ctx3 = nn.Parameter(ctx_vectors3)  # to be optimized
        self.ctx4 = nn.Parameter(ctx_vectors4)  # to be optimized
        self.ctx5 = nn.Parameter(ctx_vectors5)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, eval_only=False):

        prompts1, ctx1 = self.get_prompts(self.ctx1)
        prompts2, ctx2 = self.get_prompts(self.ctx2)
        prompts3, ctx3 = self.get_prompts(self.ctx3)
        prompts4, ctx4 = self.get_prompts(self.ctx4)
        prompts5, ctx5 = self.get_prompts(self.ctx5)

        ctxs = [ctx1, ctx2, ctx3, ctx4, ctx5]
        prompts = [prompts1, prompts2, prompts3, prompts4, prompts5]

        # if eval_only:
        #     fig = plt.figure()
        #     for i, ctx in enumerate(ctxs):
        #         writer.add_image('ctx'+str(i), ctx[i].unsqueeze(0).cpu())
        #         ax = plt.subplot(1, 5, i + 1,)
        #         # ax.set_aspect(100)
        #         plt.imshow(ctx.data[i].t().cpu(), cmap='viridis', aspect='auto')
        #     plt.show()
        #     plt.savefig('./CTX.pdf')

        return prompts, ctxs

    def get_prompts(self, ctx):
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts, ctx


CUSTOM_TEMPLATES = {
    # "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordPets": "a type of pet, a photo of a {}.",
    # "OxfordFlowers": "a photo of a {}, a type of flower.",
    "OxfordFlowers": "a type of flower, a photo of a {}.",
    "FGVCAircraft": "a type of aircraft, a photo of a {}.",
    "DescribableTextures": "a texture of {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    # "Food101": "a photo of {}, a type of food.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.cfg = cfg

    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts, ctxs = self.prompt_learner(eval_only=self.cfg.EVAL_ONLY)
        tokenized_prompts = self.tokenized_prompts


        text_features = []

        for prompt in prompts:
            text_feature = self.text_encoder(prompt, tokenized_prompts)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_features.append(text_feature)

        ctx_feas = []
        for ctx in ctxs:
            ctx_fea = self.text_encoder(ctx, tokenized_prompts, ctx=True)
            ctx_feas.append(ctx_fea)

        # fig = plt.figure()
        # for i in range(len(ctx_feas)):  # len(processed) = 17
        #     a = fig.add_subplot(1, 5, i + 1)
        #     img_plot = plt.imshow(ctx_feas[i][1].cpu().numpy())
        #     a.axis("off")
        #     # a.set_title(names[i].split('(')[0], fontsize=30)
        # plt.show()
        # # plt.savefig('ctx.jpg')

        logit_scale = self.logit_scale.exp()
        logits = 0
        for text_feature in text_features:
            logits += logit_scale * image_features @ text_feature.t()
        logits /= len(text_features)

        if self.prompt_learner.training:
            loss1 = F.cross_entropy(logits, label)
            loss2 = 0.0
            cnt = 0
            for i in range(len(ctx_feas)):
                for j in range(i + 1, len(ctx_feas)):
                    loss2 += F.cosine_similarity(ctx_feas[i][0].type(torch.float32), ctx_feas[j][0].type(torch.float32), dim=-1)
                    cnt += 1

            loss2 = 1 / loss2.shape[0] * loss2.sum() / cnt
            loss = loss1 + 5 * loss2
            return loss

        return logits


@TRAINER_REGISTRY.register()
class Prompts5(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), use all of them!"
            )
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss  = self.model(image, label)

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, model_path, epoch=None):
        names = self.get_model_names()

        for name in names:
            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights '
                'from "{}" (epoch = {})'.format(model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

