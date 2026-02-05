import torch
import torch.nn as nn
from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, Trainer
from tiny_recursive_model.transformer import init_transformer

transformer_layer = nn.TransformerEncoderLayer(
    d_model = 768,
    nhead = 8,
    dim_feedforward=1536,
    batch_first=True,
    norm_first=True,
)

trm = TinyRecursiveModel(
    dim = 768,
    # network = nn.TransformerEncoder(
    #     transformer_layer,
    #     num_layers = 2
    # ),
    network = MLPMixer1D(
        dim = 768,
        depth = 1,
        seq_len = 256,
    ),
    use_cls_token=True,
    num_refinement_blocks=2,
    num_latent_refinements=3
)
# trm.network.apply(init_transformer)

#%%
from torch.utils.data import Subset
from tiny_recursive_model.dataio import HDF

hdf_ds = HDF("/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/tmp/torgo/wavlm-basep-last/cache.hdf5",
         excluded_idx=[1685])

train_ds = Subset(hdf_ds, hdf_ds.split_idx[0])
val_ds   = Subset(hdf_ds, hdf_ds.split_idx[1])
test_ds  = Subset(hdf_ds, hdf_ds.split_idx[2])


#%%
trainer = Trainer(
    trm,
    train_ds,
    epochs = 1,
    batch_size = 16,
    cpu = False
)
trainer()
