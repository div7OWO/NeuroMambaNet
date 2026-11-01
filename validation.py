"""Checkpoint evaluation and metrics for predict.ipynb."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from checkpoints._compat import load_checkpoint


_DATASETS = {
    'BCI2A': dict(subjects=9, folds=8, classes=4, shape=(576, 1, 22, 1126),
                  checkpoint_dir='BCI_2a', prefix='bci2a'),
    'CBCIC': dict(subjects=8, folds=7, classes=2, shape=(80, 1, 12, 1536),
                 checkpoint_dir='CBCIC', prefix='CBCIC'),
}


@dataclass
class ValidationResult:
    folds: pd.DataFrame
    subjects: pd.DataFrame
    summary: pd.DataFrame
    confusion: pd.DataFrame

    def save(self, folder):
        """Export the displayed results as four small CSV files."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.folds.to_csv(folder / 'fold_results.csv', index=False)
        self.subjects.to_csv(folder / 'subject_results.csv')
        self.summary.to_csv(folder / 'summary.csv')
        self.confusion.to_csv(folder / 'confusion_matrix.csv')
        return folder.resolve()


def _read_pickle(path):
    # The local dataset files contain trusted, preprocessed tensors.
    with path.open('rb') as handle:
        return pickle.load(handle)


def _test_tensors(features, labels, shape, classes):
    features = torch.as_tensor(features).detach().cpu().float()
    labels = torch.as_tensor(labels).detach().cpu().reshape(-1)
    if tuple(features.shape) != shape or labels.numel() != shape[0]:
        raise ValueError(f'Expected X {shape}, y ({shape[0]},); got {tuple(features.shape)}, {tuple(labels.shape)}')
    if not torch.isfinite(features).all() or not torch.isfinite(labels).all():
        raise ValueError('Test data contains NaN or infinity')
    if not torch.equal(labels, labels.long().to(labels.dtype)) or not ((labels >= 0) & (labels < classes)).all():
        raise ValueError(f'Labels must be integers in 0..{classes - 1}')
    return features, labels.long()


@torch.inference_mode()
def _predict(model, loader, device, classes):
    predictions = []
    for features, _ in loader:
        logits = model(features.to(device))
        if logits.shape != (len(features), classes) or not torch.isfinite(logits).all():
            raise ValueError('Model returned an invalid shape or nonfinite logits')
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _selection(values, allowed, name):
    selected = tuple(allowed if values is None else values)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
        raise ValueError(f'{name} must be a nonempty, unique selection from {tuple(allowed)}')
    return selected


def evaluate_dataset(dataset, root=None, subjects=None, folds=None, batch_size=None, device=None):
    """Evaluate all selected saved models; no fitting or test-based model selection.

    Subjects are one-based and fold suffixes are zero-based, matching filenames.
    Accuracy/kappa are averaged across all subject–fold runs, with sample std
    (ddof=1), as in the original notebooks. Confusion matrices are row-normalized
    per run and then averaged. Max accuracy is retained as a retrospective
    descriptive statistic, not a checkpoint-selection rule.
    """
    dataset = dataset.upper()
    if dataset not in _DATASETS:
        raise ValueError(f'Unknown dataset {dataset!r}; choose BCI2A or CBCIC')
    config = _DATASETS[dataset]
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    subjects = _selection(subjects, range(1, config['subjects'] + 1), 'subjects')
    folds = _selection(folds, range(config['folds']), 'folds')
    batch_size = config['shape'][0] if batch_size is None else batch_size
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; set device="cpu"')
    # Match the deterministic cuDNN inference used by the original notebooks.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoints = {
        (subject, fold): root / 'checkpoints' / config['checkpoint_dir'] / f'{config["prefix"]}_{subject}_{fold}.pth'
        for subject in subjects for fold in folds
    }
    data_files = ([root / 'data' / 'BCI2A' / str(s) / f'bct_{s}.pkl' for s in subjects]
                  if dataset == 'BCI2A' else [root / 'data' / 'CBCIC' / 'cbcic_data.pkl'])
    missing = [str(path) for path in [*data_files, *checkpoints.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing data/checkpoint files:\n' + '\n'.join(missing))

    # CBCIC stores all eight subjects together; load this file only once.
    cbcic = _read_pickle(data_files[0]) if dataset == 'CBCIC' else None
    if cbcic is not None and (len(cbcic) != 2 or len(cbcic[0]) != 640 or len(cbcic[1]) != 640):
        raise ValueError('CBCIC expects [X, y] with 8 subjects × 80 samples in subject order')
    rows, matrices = [], []
    print(f'{dataset} | {device} | {len(subjects)} subjects × {len(folds)} folds')
    for subject in subjects:
        if dataset == 'BCI2A':
            data = _read_pickle(root / 'data' / 'BCI2A' / str(subject) / f'bct_{subject}.pkl')
            if len(data) != 4:
                raise ValueError('BCI2A expects [pool_X, pool_y, test_X, test_y]')
            features, labels = data[2], data[3]
            del data  # Release the unused training pool before inference.
        else:
            start = (subject - 1) * 80
            features, labels = cbcic[0][start:start + 80], cbcic[1][start:start + 80]
        features, labels = _test_tensors(features, labels, config['shape'], config['classes'])
        loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
        truth = labels.numpy()
        for fold in folds:
            model = load_checkpoint(checkpoints[subject, fold], device)
            predicted = _predict(model, loader, device, config['classes'])
            rows.append(dict(subject=subject, fold=fold, accuracy=float((truth == predicted).mean()),
                             kappa=float(cohen_kappa_score(truth, predicted))))
            matrices.append(confusion_matrix(truth, predicted, labels=range(config['classes']), normalize='true'))
            del model
        subject_rows = [row for row in rows if row['subject'] == subject]
        print(f"Subject {subject:02d}: accuracy={np.mean([r['accuracy'] for r in subject_rows]):.4f}, "
              f"kappa={np.mean([r['kappa'] for r in subject_rows]):.4f}")

    fold_results = pd.DataFrame(rows)
    subject_results = fold_results.groupby('subject').agg(
        folds=('fold', 'size'), accuracy=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
        max_accuracy=('accuracy', 'max'), kappa=('kappa', 'mean'),
    )
    metrics = {'Accuracy': fold_results.accuracy, 'Cohen kappa': fold_results.kappa,
               'Max subject accuracy (retrospective)': subject_results.max_accuracy}
    summary = pd.DataFrame({name: {'mean': values.mean(), 'std': values.std(ddof=1)}
                            for name, values in metrics.items()}).T
    confusion = pd.DataFrame(np.mean(matrices, axis=0),
                             index=pd.Index(range(config['classes']), name='true_label'),
                             columns=pd.Index(range(config['classes']), name='predicted_label'))
    return ValidationResult(fold_results, subject_results, summary, confusion)
