"""
Steps
1. Specify an output directory for the new run
2. See how to extract pretrained embeddings from the `infer_cli.ipynb` notebook.
3. Copy the resulting `config.yaml` file into the linear output directory
4. Run `fairseq-hydra-validate` with `common_eval.extract=[output,encoder_out]` to extract encoder outputs
5. Run the below linear training/output extraction script
"""
import argparse
import os
import yaml
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from fairseq_signals.utils.store import MemmapBatchWriter, MemmapReader, memmap_batch_process

def load_from_encoder_out(directory, split, device):
    def encoder_out_to_emb(x, device='cpu'):
        x = torch.from_numpy(np.array(x)).to(device)

        # From fairseq_signals/models/classification/ecg_transformer_classifier.py
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        return x.cpu().numpy()

    encoder_out = MemmapReader.from_header(
        os.path.join(directory, f'encoder_out_{split}.npy')
    )
    emb = memmap_batch_process(
        encoder_out,
        encoder_out_to_emb,
        256,
        progress=True,
        device=device
    )
    np.save(os.path.join(directory, f'emb_{split}.npy'), emb)

    return emb

def get_inputs(directory, splits, device):
    inputs = {}
    for split in splits:
        emb_file = os.path.join(directory, f'emb_{split}.npy')
        encoder_out_file = os.path.join(directory, f'encoder_out_{split}.npy')

        # Load from created embedding NumPy file
        if os.path.isfile(emb_file):
            inputs[split] = np.load(emb_file)
        # Load from extracted encoder out NumPy memmap file
        elif os.path.isfile(encoder_out_file):
            inputs[split] = load_from_encoder_out(directory, split, device)
        else:
            raise FileNotFoundError(
                f"No file '{encoder_out_file}'. Extract the encoder outputs for the {split} subset."
            )

    return inputs

def get_loaders(directory, manifest_file, splits, device):
    with open(os.path.join(directory, 'config.yaml'), "r") as f:
        config = yaml.safe_load(f)

    manifest_dir = config['task']['data']
    label_file = config['task']['label_file']

    # Load manifests and align by sample ID ('idx')
    manifest_all = pd.read_csv(manifest_file, low_memory=False)
    manifests = {split: pd.read_csv(os.path.join(manifest_dir, f'{split}.tsv'), sep='\t', index_col='Unnamed: 0') for split in splits}
    for split in manifests:
        manifest = manifests[split]
        manifest.index.name = 'file'
        manifest.reset_index(inplace=True)
        manifest['save_file'] = manifest['file'].str.split('/').str[-1].replace('_\d+\.mat$', '.mat', regex=True)
        manifests[split] = manifest.merge(manifest_all[['save_file', 'idx']], on='save_file', how='left')

    # Load targets
    label_dir = os.path.dirname(label_file)
    label_def = pd.read_csv(os.path.join(label_dir, "label_def.csv"))
    y = np.load(label_file)
    labels = {split: y[manifests[split]["idx"].values] for split in splits}

    # Load inputs (model embeddings)
    inputs = get_inputs(directory, splits, device)

    not_null = ~np.isnan(inputs['test']).any(axis=1)
    inputs['test'] = inputs['test'][not_null]
    labels['test'] = labels['test'][not_null]

    # Define data loaders
    datasets = {split: LinearDataset(inputs[split], labels[split]) for split in inputs.keys()}
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=256,
            shuffle=(split == 'train'),
        ) for split in datasets.keys()
    }

    return inputs, labels, loaders

def compute_multilabel_auroc(y_true, y_scores):
    """
    Compute the average AUROC for multilabel classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels in binary indicator format.
    y_scores : array-like
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.

    Returns
    -------
    float
        The average AUROC across all labels.
    """
    n_classes = y_true.shape[1]
    aurocs = []
    for i in range(n_classes):
        auroc = roc_auc_score(y_true[:, i], y_scores[:, i])
        aurocs.append(auroc)

    return np.mean(aurocs)

def compute_multilabel_auprc(y_true, y_scores):
    """
    Compute the average AUPRC for multilabel classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels in binary indicator format.
    y_scores : array-like
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.

    Returns
    -------
    float
        The average AUPRC across all labels.
    """
    n_classes = y_true.shape[1]
    auprcs = []

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        auprc = auc(recall, precision)
        auprcs.append(auprc)

    return np.mean(auprcs)

class LinearClassifier(nn.Module):
    def __init__(self, encoder_embed_dim, num_labels):
        super().__init__()
        self.proj = nn.Linear(encoder_embed_dim, num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x):
        logits = self.proj(x)

        return logits

class LinearDataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = inputs
        self.labels = torch.from_numpy(labels).float()

    def __getitem__(self, index: int):
        return torch.from_numpy(self.inputs[index]).float(), self.labels[index]

    def __len__(self):
        return len(self.inputs)

class Trainer:
    def __init__(
        self,
        model,
        loaders,
        optimizer,
        device,
        checkpoint_path,
        n_epochs=10,
        save_every_n=1,
        from_checkpoint=None,
    ):
        model = model.to(device)

        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.device = device
        self.n_epochs = n_epochs
        self.checkpoint_path = checkpoint_path
        self.save_every_n = save_every_n
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_losses = []
        self.valid_losses = []
        self.valid_aurocs = []
        self.valid_auprcs = []
        self.best_auprc = 0.0
        self.prev_epoch = 0

        if from_checkpoint:
            self.load_checkpoint(from_checkpoint)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_losses = checkpoint['train_losses']
        self.valid_losses = checkpoint['valid_losses']
        self.valid_aurocs = checkpoint['valid_aurocs']
        self.valid_auprcs = checkpoint['valid_auprcs']
        self.best_auprc = checkpoint['best_auprc']
        self.prev_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{path}' (epoch {self.prev_epoch})")

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': self.prev_epoch + epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_auprc': self.best_auprc,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'valid_aurocs': self.valid_aurocs,
            'valid_auprcs': self.valid_auprcs,
        }
        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            print(f"New best checkpoint saved with AUPRC: {self.best_auprc:.4f}")

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels in self.loaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(self.loaders['train'])
            self.train_losses.append(avg_train_loss)

            # Validation phase
            valid_metrics = self.evaluate()
            valid_loss, valid_auroc, valid_auprc = \
                valid_metrics['loss'], valid_metrics['auroc'], valid_metrics['auprc']
            self.valid_losses.append(valid_loss)
            self.valid_aurocs.append(valid_auroc)
            self.valid_auprcs.append(valid_auprc)

            if (epoch % self.save_every_n == self.save_every_n - 1):
                if valid_auprc > self.best_auprc:
                    self.best_auprc = valid_auprc
                    self.save_checkpoint(epoch, is_best=True)

            print(
                f'Epoch {self.prev_epoch + epoch + 1} - '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {valid_loss:.4f}, '
                f'Val AUROC: {valid_auroc:.4f}, '
                f'Val AUPRC: {valid_auprc:.4f}'
            )

    def _evaluate(self, split='valid'):
        self.model.eval()
        valid_loss = 0.0
        all_targets = []
        all_logits = []

        with torch.no_grad():
            for inputs, labels in self.loaders[split]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                valid_loss += loss.item()

                all_targets.append(labels.detach().cpu())
                all_logits.append(logits.detach().cpu())

        avg_valid_loss = valid_loss / len(self.loaders[split])
        all_targets = torch.vstack(all_targets)
        all_logits = torch.vstack(all_logits)

        return all_logits, all_targets, avg_valid_loss

    def evaluate(self, split='valid'):
        logits, targets, avg_valid_loss = self._evaluate(split=split)
        pred = torch.sigmoid(logits).numpy()
        valid_auroc = compute_multilabel_auroc(targets, pred)
        valid_auprc = compute_multilabel_auprc(targets, pred)

        return {'loss': avg_valid_loss, 'auroc': valid_auroc, 'auprc': valid_auprc}

    def save_outputs(self, save_dir, split='valid'):
        logits, targets, _ = self._evaluate(split=split)
        logits, targets = logits.numpy(), targets.numpy()

        logit_writer = MemmapBatchWriter(
            os.path.join(save_dir, f'checkpoint{self.prev_epoch}_outputs_{split}.npy'),
            logits.shape,
            dtype=logits.dtype,
        )
        logit_writer(logits)

        target_writer = MemmapBatchWriter(
            os.path.join(save_dir, f'checkpoint{self.prev_epoch}_targets_{split}.npy'),
            targets.shape,
            dtype=targets.dtype,
        )
        target_writer(targets)

def plot(trainer):
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(trainer.valid_aurocs, label='Validation AUROC')
    plt.title('Validation AUROC per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(trainer.valid_auprcs, label='Validation AUPRC')
    plt.title('Validation AUPRC per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('AUPRC')
    plt.legend()
    plt.show()

def main(args):
    inputs, labels, loaders = get_loaders(
        args.directory,
        args.manifest_file,
        args.splits,
        args.device,
    )
    clf = LinearClassifier(inputs['train'].shape[1], labels["train"].shape[1])
    optimizer = torch.optim.Adam(
        clf.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
    )
    trainer = Trainer(
        clf,
        loaders,
        optimizer,
        args.device,
        os.path.join(args.directory, 'linear.pt'),
        n_epochs=args.n_epochs,
        save_every_n=10,
    )
    trainer.train()
    plot(trainer)

    # Load best checkpoint
    trainer.load_checkpoint(os.path.join(args.directory, 'linear.pt'))

    print('\nFinal validation metrics')
    print(trainer.evaluate(split='valid'))

    print('\nFinal test metrics')
    print(trainer.evaluate(split='test'))

    checkpoint_metrics = pd.DataFrame({
        'train_loss': trainer.train_losses,
        'valid_loss': trainer.valid_losses,
        'valid_auroc': trainer.valid_losses,
        'valid_auprc':  trainer.valid_auprcs,
    })
    checkpoint_metrics.index.name = 'checkpoint'
    checkpoint_metrics.index += 1
    checkpoint_metrics.to_csv(os.path.join(args.directory, 'train.csv'))

    # Save outputs
    trainer.save_outputs(args.directory, split='valid')
    trainer.save_outputs(args.directory, split='test')

    return inputs, labels, trainer

def get_parser():
    """
    Define the command-line arguments needed for the script.

    Returns
    -------
    argparse.ArgumentParser
        Returns an ArgumentParser object with configured arguments.
    """
    parser = argparse.ArgumentParser(description="Command-line interface for the deep learning pipeline.")
    
    # Required arguments
    parser.add_argument(
        '--directory', 
        type=str, 
        required=True, 
        help='The base directory where input files are located and outputs should be saved.'
    )
    parser.add_argument(
        '--manifest_file', 
        type=str, 
        required=True, 
        help='Path to the manifest file that contains data references.'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda:0', 
        help='The device on which to perform computations.'
    )
    parser.add_argument(
        '--splits', 
        nargs='+', 
        default=['train', 'valid', 'test'], 
        help='The data splits to process. Default is ["train", "valid", "test"].'
    )
    parser.add_argument(
        '--n_epochs', 
        type=int, 
        default=5000, 
        help='Number of training epochs. Default is 5000.'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-5, 
        help='Learning rate for the optimizer. Default is 1e-5.'
    )
    parser.add_argument(
        '--betas', 
        type=float, 
        nargs=2, 
        default=(0.9, 0.98), 
        help='Betas for Adam optimizer. Default is (0.9, 0.98).'
    )
    parser.add_argument(
        '--eps', 
        type=float, 
        default=1e-08, 
        help='Epsilon for Adam optimizer. Default is 1e-08.'
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    inputs, labels, trainer = main(args)
