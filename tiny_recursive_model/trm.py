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

class SelectFirstToken(Module):
    """Extracts the first token from a sequence (for CLS token)"""
    def forward(self, x):
        return x[:, 0]

class TinyRecursiveModel(Module):
    def __init__(
        self,
        *,
        dim,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper - 1 output refinement per N latent refinements
        halt_loss_weight = 1.,
        use_cls_token = False
    ):
        super().__init__()
        assert num_refinement_blocks > 1
        self.use_cls_token = use_cls_token

        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        # CLS token for classification
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(dim) * 1e-2)

        self.network = network

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        # prediction heads

        # CLS token uses first position, otherwise mean pooling
        # Create separate instances for each head to avoid module sharing issues
        def make_pool_layer():
            return SelectFirstToken() if use_cls_token else Reduce('b n d -> b d', 'mean')

        self.to_pred = nn.Sequential(
            make_pool_layer(),
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

    @property
    def device(self):
        return next(self.parameters()).device

    def prepend_cls_token(self, seq):
        """Prepend CLS token to sequence if use_cls_token is enabled"""
        if not self.use_cls_token:
            return seq

        batch = seq.shape[0]
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = batch)
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

        for i in range(self.num_latent_refinements):
            combined = outputs + latents + inputs

            if self.training and hasattr(self, '_debug_refinement'):
                print(f"  Latent refine {i}: combined_norm={combined.norm().item():.2f}", end='')

            latents = self.network(combined)

            if self.training and hasattr(self, '_debug_refinement'):
                print(f" -> latents_norm={latents.norm().item():.2f}")

        combined_out = outputs + latents

        if self.training and hasattr(self, '_debug_refinement'):
            print(f"  Output refine: combined_norm={combined_out.norm().item():.2f}", end='')

        outputs = self.network(combined_out)

        if self.training and hasattr(self, '_debug_refinement'):
            print(f" -> outputs_norm={outputs.norm().item():.2f}")

        return outputs, latents

    def deep_refinement(
        self,
        inputs,    # (b n d)
        outputs,   # (b n d)
        latents,   # (b n d)
    ):

        for step in range_from_one(self.num_refinement_blocks):

            # only last round of refinement receives gradients

            is_last = step == self.num_refinement_blocks
            context = torch.no_grad if not is_last else nullcontext

            with context():
                outputs, latents = self.refine_latent_then_output_once(inputs, outputs, latents)

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
        # prepend CLS token if enabled
        seq = self.prepend_cls_token(seq)

        outputs, latents = self.deep_refinement(seq, outputs, latents)

        pred = self.to_pred(outputs)

        halt_logits = self.to_halt_pred(outputs)

        halt_prob = halt_logits.sigmoid()

        outputs, latents = outputs.detach(), latents.detach()

        return_package = (outputs, latents, pred, halt_prob)

        if not exists(labels):
            return return_package

        # calculate loss if labels passed in

        loss = F.binary_cross_entropy_with_logits(pred, labels.float(), reduction = 'none')

        is_all_correct = pred == labels

        halt_loss = F.binary_cross_entropy_with_logits(halt_logits, is_all_correct.float(), reduction = 'none')

        # total loss and loss breakdown

        total_loss = (
            loss +
            halt_loss * self.halt_loss_weight
        )

        losses = (loss, halt_loss)

        return total_loss.sum(), losses, *return_package

