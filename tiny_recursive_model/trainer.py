from __future__ import annotations

from pathlib import Path
from pprint import pformat

import accelerate.tracking
import torch
from tiny_recursive_model.eval import evaluate
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from einops import pack, unpack

from accelerate import Accelerator

# ema - apparently greatly help;.ed with results

from ema_pytorch import EMA

from tiny_recursive_model.dataio import padded_batch
from tiny_recursive_model.trm import TinyRecursiveModel

from adam_atan2_pytorch import MuonAdamAtan2

from x_transformers import Encoder, Decoder

# helpers

def exists(v):
    return v is not None

def range_from_one(n):
    return range(1, n + 1)

def is_empty(t):
    return t.numel() == 0

# trainer

class Trainer(Module):
    def __init__(
        self,
        model: TinyRecursiveModel | Module,
        dataset: Dataset,
        val: Dataset | None = None,
        test: Dataset | None = None,
        optim_klass = AdamW,
        optim: Optimizer | None = None,
        learning_rate = 1e-4,
        muon_learning_rate = 1e-3,
        weight_decay = 1.,
        batch_size = 16,
        epochs = 2,
        halt_prob_thres = 0.5,
        max_recurrent_steps = 12, # N_sup in paper
        warmup_steps = 2000,
        ema_decay_rate = 0.999,
        switch_ema_every = 10000,           # switch ema https://arxiv.org/abs/2402.09240
        accelerate_kwargs: dict = dict(),
        cpu = False,
        output_dir: Path | None = None,
    ):
        super().__init__()

        self.accelerator  = Accelerator(**accelerate_kwargs, cpu = cpu)

        self.batch_size = batch_size

        self.epochs = epochs

        # data

        self.dataset = dataset

        self.dataloader = dataloader = DataLoader(self.dataset,
                                                  batch_size = self.batch_size,
                                                  shuffle = True,
                                                  collate_fn = padded_batch)

        if val:
            self.val_dataloader = DataLoader(val, batch_size = self.batch_size, shuffle = False, collate_fn = padded_batch)

        if test:
            self.test_dataloader = DataLoader(test, batch_size = self.batch_size, shuffle = False, collate_fn = padded_batch)

        if not exists(optim):
            if isinstance(model.network, (Encoder, Decoder)):
                optim = MuonAdamAtan2(
                    model.network.muon_parameters(),
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    muon_lr = muon_learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )
            else:
                optim = optim_klass(
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )

        self.optim = optim

        # scheduler

        self.scheduler = LambdaLR(self.optim, lambda step: min((step + 1) / warmup_steps, 1.0))

        # model

        self.model = model

        # ema model

        self.ema_model = None

        if self.accelerator.is_main_process:
            self.ema_model = EMA(
                model,
                beta = ema_decay_rate,
                update_model_with_ema_every = switch_ema_every,
                forward_method_names = ('predict',)
            )

        # recurrent and act related variables

        self.halt_prob_thres = halt_prob_thres

        self.max_recurrent_steps = max_recurrent_steps

        # prepare maybe distributed

        self.model, self.optim, self.dataloader, self.scheduler = self.accelerator.prepare(self.model, self.optim, self.dataloader, self.scheduler)

        # move EMA model to the same device as the prepared model
        if self.ema_model is not None:
            self.ema_model.to(self.accelerator.device)

        self.output_dir = output_dir

        self.tracker = accelerate.tracking.TensorBoardTracker(output_dir.name, output_dir.parent)

    def forward(self):

        for epoch in range_from_one(self.epochs):

            for batch, (dataset_input, dataset_output) in enumerate(self.dataloader):

                num_batches = len(self.dataloader)

                outputs, latents = self.model.get_initial()

                for recurrent_step in range_from_one(self.max_recurrent_steps):

                    loss, (main_loss, halt_loss), outputs, latents, pred, halt = self.model(dataset_input, outputs, latents, labels = dataset_output)

                    self.accelerator.print(f'[Epoch {epoch} Batch {batch}/{num_batches} ({recurrent_step} / {self.max_recurrent_steps})] loss: {main_loss.mean().item():.3f} | halt loss: {halt_loss.mean().item():.3f}')

                    self.accelerator.backward(loss)

                    self.optim.step()
                    self.optim.zero_grad()

                    self.scheduler.step()

                    if self.accelerator.is_main_process:
                        self.ema_model.update()

                    # handle halting

                    halt_mask = halt >= self.halt_prob_thres

                    if not halt_mask.any():
                        continue

                    outputs = outputs[~halt_mask]
                    latents = latents[~halt_mask]
                    dataset_input = dataset_input[~halt_mask]
                    dataset_output = dataset_output[~halt_mask]

                    if is_empty(outputs):
                        break

            if self.val_dataloader:
                self.accelerator.print(f'--- Epoch {epoch} validation ---')

                results = evaluate(self.model,
                                   self.val_dataloader,
                                   device=self.accelerator.device,
                                   ext='val',
                                   threshold=self.halt_prob_thres)

                self.tracker.log(results, epoch)
                self.accelerator.print(results)

        if self.test_dataloader:
            self.accelerator.print(f'--- Test evaluation ---')

            results = evaluate(self.model,
                               self.test_dataloader,
                               device=self.accelerator.device,
                               ext='test',
                               threshold=self.halt_prob_thres)

            self.tracker.log(results, self.epochs + 1)

            self.accelerator.print(results)

        self.accelerator.print('complete')

        if self.accelerator.is_main_process:
            self.ema_model.copy_params_from_ema_to_model()
