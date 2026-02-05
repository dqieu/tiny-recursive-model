import argparse

import torch
import torch.nn as nn
from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, Trainer
from tiny_recursive_model.transformer import init_transformer

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description="Train Tiny Recursive Model")
    parser.add_argument('--data-path', '-p', type=str, required=True, help='Path to the HDF5 dataset')
    parser.add_argument('--excluded-idx', '-e', type=int, nargs='*', default=[], help='Indices to exclude from the dataset')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--output-path', '-o', type=str, required=True, help='Path to save the trained model')
    args = parser.parse_args()

    # transformer_layer = nn.TransformerEncoderLayer(
    #     d_model = 768,
    #     nhead = 8,
    #     dim_feedforward=1536,
    #     batch_first=True,
    #     norm_first=True,
    # )

    trm = TinyRecursiveModel(
        dim=768,
        # network = nn.TransformerEncoder(
        #     transformer_layer,
        #     num_layers = 2
        # ),
        network=MLPMixer1D(
            dim=768,
            depth=1,
            seq_len=256,
        ),
        use_cls_token=True,
        num_refinement_blocks=2,
        num_latent_refinements=3
    )
    # trm.network.apply(init_transformer)

    from torch.utils.data import Subset
    from tiny_recursive_model.dataio import HDF

    # "/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/tmp/torgo/wavlm-basep-last/cache.hdf5"
    hdf_ds = HDF(args.data_path,
                 excluded_idx=args.excluded_idx,)

    train_ds = Subset(hdf_ds, hdf_ds.split_idx[0])
    val_ds = Subset(hdf_ds, hdf_ds.split_idx[1])
    test_ds = Subset(hdf_ds, hdf_ds.split_idx[2])

    trainer = Trainer(
        trm,
        train_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cpu=False
    )
    trainer()

    torch.save(trm.state_dict(), args.output_path)