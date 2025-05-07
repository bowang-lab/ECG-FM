"""
Convert the encoder_out precursors into a final embedding form (see `infer_cli.ipynb` notebook for extraction). See the `infer_quickstart.ipynb` for visualization.
"""
import os

import numpy as np

import torch

from fairseq_signals.utils.store import (
    MemmapReader,
    memmap_batch_process,
)

def main(args):
    def encoder_out_to_emb(x, device='cpu'):
        x = torch.from_numpy(np.array(x)).to(device)

        # From fairseq_signals/models/classification/ecg_transformer_classifier.py
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        return x.cpu().numpy()

    encoder_out = MemmapReader.from_header(
        os.path.join(args.directory, f'encoder_out_{args.split}.npy')
    )
    emb = memmap_batch_process(
        encoder_out,
        encoder_out_to_emb,
        256,
        progress=True,
        device=args.device
    )
    np.save(os.path.join(args.directory, f'emb_{args.split}.npy'), emb)

def get_parser():
    """
    Define the command-line arguments needed for the script.

    Returns
    -------
    argparse.ArgumentParser
        Returns an ArgumentParser object with configured arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory', 
        type=str, 
        required=True, 
        help='The base directory where the extracted saliency files are located.'
    )
    parser.add_argument(
        '--split', 
        type=str, 
        required=True, 
        help='The data split to process.'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0', 
        help='The device on which to perform computations.'
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
