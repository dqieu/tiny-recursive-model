from __future__ import annotations
from contextlib import nullcontext
from pprint import pformat

import torch
from torch import nn, cat, arange, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Reduce, Rearrange

# network related

from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return t.numel() == 0

def range_from_one(n):
    return range(1, n + 1)

# classes

class SelectCLSToken(Module):
    """Extracts CLS tokens and then mean pool them"""
    def __init__(self, num_tokens = 1):
        super().__init__()
        self.num_tokens = num_tokens

    def forward(self, x):
        return x[:, :self.num_tokens].mean(dim = 1)

class TinyRecursiveModel(Module):
    def __init__(
        self,
        *,
        dim,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper - 1 output refinement per N latent refinements
        halt_loss_weight = 1.,
        pos_weight = 1
    ):
        super().__init__()
        assert num_refinement_blocks > 1

        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        # CLS token for classification
        self.num_cls_tokens = num_refinement_blocks + 1
        cls_token_weight = torch.randn((self.num_cls_tokens, dim))
        self.cls_token = nn.Parameter(cls_token_weight * 1e-2)


        self.network = network

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        # prediction heads

        self.to_pred = nn.Sequential(
            SelectCLSToken(self.num_refinement_blocks),
            nn.Linear(dim, 1, bias = False),
            Rearrange('... 1 -> ...')
        )

        self.to_halt_pred = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, 1, bias = False),
            Rearrange('... 1 -> ...')
        )

        self.halt_loss_weight = halt_loss_weight

        # init
        nn.init.zeros_(self.to_halt_pred[1].weight)

        self.pos_weight = torch.tensor(pos_weight)

    @property
    def device(self):
        return next(self.parameters()).device

    def prepend_cls_token(self, seq, i = 0):
        """Prepend CLS token to sequence if use_cls_token is enabled"""
        if not self.use_cls_token:
            return seq

        batch = seq.shape[0]
        cls_tokens = repeat(self.cls_token[i], 'd -> b 1 d', b = batch)
        return cat([cls_tokens, seq], dim = 1)

    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed

        return outputs, latents

    def refine_latent_then_output_once(
        self,
        inputs,     # (b n d)
        outputs,    # (b n d)
        latents,    # (b n d)
    ):

        # so it seems for this work, they use only one network
        # the network learns to refine the latents if input is passed in, otherwise it refines the output

        for _ in range(self.num_latent_refinements):

            latents = self.network(outputs + latents + inputs)

        outputs = self.network(outputs + latents)

        return outputs, latents

    def deep_refinement(
        self,
        inputs,    # (b n d)
        outputs,   # (d)
        latents,   # (d)
    ):
        batch = inputs.shape[0]

        clss = repeat(self.cls_token, 'l d -> b l d', b = batch)

        inputs, _ = pack([clss[:, 0], inputs], 'b * d')

        for step in range_from_one(self.num_refinement_blocks):
            # only last round of refinement receives gradients

            is_last = step == self.num_refinement_blocks
            context = torch.no_grad if not is_last else nullcontext

            with context():
                outputs, latents = self.refine_latent_then_output_once(inputs, outputs, latents)

                inputs, _ = pack([clss[:, step], inputs], 'b * d')

                init_output, init_latent = self.get_initial()

                init_output = repeat(init_output, 'd -> b 1 d', b = batch)
                init_latent = repeat(init_latent, 'd -> b 1 d', b = batch)

                outputs, _ = pack([init_output, outputs], 'b * d')
                latents, _ = pack([init_latent, latents], 'b * d')


        return outputs, latents

    @torch.no_grad()
    def predict(
        self,
        seq,
        halt_prob_thres = 0.5,
        max_deep_refinement_steps = 12
    ):
        batch = seq.shape[0]

        # prepend CLS token if enabled
        inputs = self.prepend_cls_token(seq)
        # initial outputs and latents

        outputs, latents = self.get_initial()

        # active batch indices, the step it exited at, and the final output predictions

        active_batch_indices = arange(batch, device = self.device, dtype = torch.long)

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        for step in range_from_one(max_deep_refinement_steps):
            is_last = step == max_deep_refinement_steps

            outputs, latents = self.deep_refinement(inputs, outputs, latents)

            halt_prob = self.to_halt_pred(outputs).sigmoid()

            should_halt = (halt_prob >= halt_prob_thres) | is_last

            if not should_halt.any():
                continue

            # append to exited predictions

            pred = self.to_pred(outputs[should_halt])
            preds.append(pred)

            # append the step at which early halted

            exited_step_indices.extend([step] * should_halt.sum().item())

            # append indices for sorting back

            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            # ready for next round

            inputs = inputs[~should_halt]
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
            active_batch_indices = active_batch_indices[~should_halt]

            if is_empty(outputs):
                break

        preds = cat(preds)
        exited_step_indices = tensor(exited_step_indices, device=self.device)

        exited_batch_indices = cat(exited_batch_indices)
        sort_indices = exited_batch_indices.argsort(dim = -1)

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(
        self,
        seq,
        outputs,
        latents,
        labels = None
    ):

        outputs, latents = self.deep_refinement(seq, outputs, latents)

        pred = self.to_pred(outputs)

        # halt loss goes to all but cls tok
        cls, toks = outputs[:, :self.num_cls_tokens], outputs[:, self.num_cls_tokens:]
        cls = cls.detach()
        outputs = torch.cat([cls, toks], dim = 1)

        halt_logits = self.to_halt_pred(outputs)

        halt_prob = halt_logits.sigmoid()

        outputs, latents = outputs.detach(), latents.detach()

        return_package = (outputs, latents, pred, halt_prob)

        if not exists(labels):
            return return_package

        # calculate loss if labels passed in

        # change if multiclass
        loss = F.binary_cross_entropy_with_logits(pred, labels.float(), reduction = 'none', pos_weight=self.pos_weight)

        is_all_correct = (pred.sigmoid() > .5).long() == labels

        halt_loss = F.binary_cross_entropy_with_logits(halt_logits, is_all_correct.float(), reduction = 'none')

        # total loss and loss breakdown

        total_loss = (
            loss +
            halt_loss * self.halt_loss_weight
        )

        losses = (loss, halt_loss)

        return total_loss.sum(), losses, *return_package

