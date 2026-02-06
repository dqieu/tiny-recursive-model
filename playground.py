import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tiny_recursive_model import TinyRecursiveModel, MLPMixer1D, Trainer
from tiny_recursive_model.transformer import init_transformer
from torch.utils.data import Subset
from tiny_recursive_model.dataio import HDF


def get_config_dict(obj):
    return {k: str(v) for k, v in vars(obj).items()}

if __name__ == '__main__':
    # Parsing arguments

    parser = argparse.ArgumentParser(description="Train Tiny Recursive Model")

    parser.add_argument('--data-path', '-p', type=str,
                        default="/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/tmp/torgo/wavlm-basep-last/cache.hdf5",
                        help='Path to the HDF5 dataset')

    parser.add_argument('--excluded-idx', '-e', type=int, nargs='*', default=[1685], help='Indices to exclude from the dataset')

    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size for training')

    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    parser.add_argument('--output-dir', '-o', type=str, default='exps/0', help='Path to save the trained model')

    args = parser.parse_args()

    args_cfg = get_config_dict(args)

    transformer_layer = nn.TransformerEncoderLayer(
        d_model=768,
        nhead=8,
        dim_feedforward=3072,
        batch_first=True,
        norm_first=True,
    )
    network = nn.Sequential(
        nn.TransformerEncoder(
            transformer_layer,
            num_layers=2
        ),
        nn.LayerNorm(768)
    )

    trm = TinyRecursiveModel(
        dim=768,
        network = network,
        # network=MLPMixer1D(
        #     dim=768,
        #     depth=1,
        #     seq_len=256,
        # ),
        use_cls_token=False,
        num_refinement_blocks=3,
        num_latent_refinements=3
    )

    trm.network.apply(init_transformer)

    trm_cfg = get_config_dict(trm)

    hdf_ds = HDF(args.data_path, excluded_idx=args.excluded_idx, length = 511)

    train_ds = Subset(hdf_ds, hdf_ds.split_idx[0])

    val_ds = Subset(hdf_ds, hdf_ds.split_idx[1])

    test_ds = Subset(hdf_ds, hdf_ds.split_idx[2])

    output_dir = Path(args.output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)


    trainer = Trainer(
        trm,
        train_ds,
        val_ds,
        test_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cpu=False,
        output_dir=output_dir
    )

    trainer_cfg = get_config_dict(trainer)

    trainer.accelerator.init_trackers("trm", config=args_cfg | trm_cfg | trainer_cfg)

    trainer()

    torch.save(trm.state_dict(), output_dir / "trm_model.pt")