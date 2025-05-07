"""
Convert the saliency map precursors into a more final form (see `infer_cli.ipynb` notebook for extraction). See the `infer_quickstart.ipynb` for visualization.
"""
import argparse
import os

import numpy as np

import torch

from fairseq_signals.utils.store import MemmapReader

def main(args):
    saliency = MemmapReader.from_header(
        os.path.join(args.directory, f'saliency_{args.split}.npy')
    )

    # Consider attention weights of the final layer
    attn = saliency[:, -1]
    attn = torch.from_numpy(np.array(attn)).to(args.device)
    attn_max = attn.max(axis=2).values.squeeze().cpu().detach().numpy()
    np.save(os.path.join(args.directory, f'attn_max_{args.split}.npy'), attn_max)

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
