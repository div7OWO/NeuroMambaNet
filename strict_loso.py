"""Strict LOSO: one held-out test subject, one validation block per fit.

The notebook supplies the model factory; this module owns splitting, training,
logging and artifact storage. Test data never participates in model selection.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import logging
import math
import pickle
import random
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


@dataclass
class TrainConfig:
    data_dir: str = "data"
    output_dir: str = "checkpoints"
    model_name: str = "model"
    test_subjects: tuple = tuple(range(1, 10))
    n_subjects: int = 9
    samples_per_subject: int = 576
    batch_size: int = 288
    max_epochs: int = 600
    patience: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    # None means all n_subjects - 1 blocks; (0,) is useful for a smoke run.
    validation_blocks: tuple | None = None

    def validate(self):
        for name in ("samples_per_subject", "batch_size", "max_epochs", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.n_subjects < 3 or self.num_workers < 0:
            raise ValueError("Need >= 3 subjects and num_workers >= 0")
        if not self.test_subjects or len(set(self.test_subjects)) != len(self.test_subjects):
            raise ValueError("test_subjects must be nonempty and unique")
        if any(s < 1 or s > self.n_subjects for s in self.test_subjects):
            raise ValueError("test_subjects must be in 1..n_subjects")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.seed < 0:
            raise ValueError("Invalid optimizer configuration or seed")
        blocks = self.blocks
        if not blocks or len(set(blocks)) != len(blocks):
            raise ValueError("validation_blocks must be nonempty and unique")
        if any(b < 0 or b >= self.n_subjects - 1 for b in blocks):
            raise ValueError("validation_blocks are zero-based indices in 0..n_subjects-2")

    @property
    def blocks(self):
        return tuple(range(self.n_subjects - 1)) if self.validation_blocks is None else self.validation_blocks


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Custom CUDA kernels can still be nondeterministic.


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_logger(folder):
    logger = logging.getLogger(f"strict_loso.{folder.name}.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(folder / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_subject_data(config, subject):
    path = Path(config.data_dir) / str(subject) / f"bct_{subject}.pkl"
    # These pickle files are locally supplied trusted training data.
    with path.open("rb") as handle:
        arrays = pickle.load(handle)
    if len(arrays) != 4:
        raise ValueError(f"{path}: expected [pool_x, pool_y, test_x, test_y]")
    pool_x, pool_y, test_x, test_y = [torch.as_tensor(x).detach().cpu() for x in arrays]
    pool_x, test_x = pool_x.float(), test_x.float()
    pool_y, test_y = pool_y.reshape(-1), test_y.reshape(-1)
    expected = (config.n_subjects - 1) * config.samples_per_subject
    if len(pool_x) != expected or len(pool_y) != expected:
        raise ValueError(f"{path}: expected {expected} pool samples, got {len(pool_x)}/{len(pool_y)}")
    if not len(test_x) or len(test_x) != len(test_y):
        raise ValueError(f"{path}: test data/labels must have equal nonzero lengths")
    if pool_x.shape[1:] != test_x.shape[1:]:
        raise ValueError(f"{path}: pool/test feature shapes differ")
    for name, labels in (("pool", pool_y), ("test", test_y)):
        if not torch.isfinite(labels).all() or (labels < 0).any() or not torch.equal(labels, labels.long().to(labels.dtype)):
            raise ValueError(f"{path}: {name} labels must be finite nonnegative integers")
    return TensorDataset(pool_x, pool_y.long()), TensorDataset(test_x, test_y.long()), path


def split_indices(pool_size, block, block_size):
    """Same contiguous validation slices as the original notebook, without copies."""
    start, stop = block * block_size, (block + 1) * block_size
    if start < 0 or stop > pool_size:
        raise ValueError("Validation block falls outside the training pool")
    train_indices = list(range(start)) + list(range(stop, pool_size))
    validation_indices = list(range(start, stop))
    return train_indices, validation_indices


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss encountered")
            if training:
                loss.backward()
                optimizer.step()
            count = labels.numel()
            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += count
    if total_samples == 0:
        raise ValueError("Empty data loader")
    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(path, model, optimizer, epoch, best_epoch, best_accuracy, config, split):
    # state_dict avoids pickling the model class itself.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "config": asdict(config),
        "split": split,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def plot_history(history_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(history_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric, label in zip(axes, ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
        for subset in ("train", "val"):
            ax.plot(epochs, [float(row[f"{subset}_{metric}"]) for row in rows], label=subset)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_ylim(0, 1)
    try:
        fig.savefig(output_path, dpi=160)
    finally:
        plt.close(fig)


def train_fold(model_factory, config, pool, test_data, subject, block, run_dir, device, source_path):
    fold_dir = run_dir / f"test_subject_{subject:02d}" / f"val_block_{block + 1:02d}"
    fold_dir.mkdir(parents=True, exist_ok=False)
    logger = make_logger(fold_dir)
    train_ids, val_ids = split_indices(len(pool), block, config.samples_per_subject)
    # Stable per-fit seed makes a selected fold reproducible without running earlier folds.
    fold_seed = config.seed + (subject - 1) * (config.n_subjects - 1) + block
    split = {
        "test_subject": subject, "validation_block_zero_based": block,
        "validation_subject": None,
        "subject_order_note": "Block subject identities are not provided by the pickle; do not infer them.",
        "data_file": str(source_path.resolve()),
        "validation_slice": [val_ids[0], val_ids[-1] + 1],
        "train_indices": train_ids, "validation_indices": val_ids,
        "train_samples": len(train_ids), "val_samples": len(val_ids), "test_samples": len(test_data),
        "fold_seed": fold_seed,
    }
    started = time.perf_counter()
    history_path = fold_dir / "history.csv"
    try:
        seed_everything(fold_seed)
        write_json(fold_dir / "config.json", {"training": asdict(config), "split": split, "device": str(device)})
        write_json(fold_dir / "result.json", {"status": "running"})
        options = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=device.type == "cuda")
        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = DataLoader(Subset(pool, train_ids), shuffle=True, generator=generator, **options)
        val_loader = DataLoader(Subset(pool, val_ids), shuffle=False, **options)
        test_loader = DataLoader(test_data, shuffle=False, **options)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_accuracy, best_epoch, stale_epochs = -math.inf, 0, 0
        logger.info("Start | test subject=%s | validation block=%s | seed=%s | device=%s | samples=%s/%s/%s",
                    subject, block + 1, fold_seed, device, len(train_ids), len(val_ids), len(test_data))
        fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate",
                  "best_val_accuracy", "best_epoch", "epochs_without_improvement", "is_best", "epoch_seconds", "elapsed_seconds"]
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            for epoch in range(1, config.max_epochs + 1):
                epoch_started = time.perf_counter()
                train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
                val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
                improved = val_accuracy > best_accuracy
                if improved:
                    best_accuracy, best_epoch, stale_epochs = val_accuracy, epoch, 0
                    save_checkpoint(fold_dir / "best.pt", model, optimizer, epoch, best_epoch, best_accuracy, config, split)
                else:
                    stale_epochs += 1
                # Always retains the last completed epoch, including when interrupted later.
                save_checkpoint(fold_dir / "last.pt", model, optimizer, epoch, best_epoch, best_accuracy, config, split)
                row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_accuracy,
                           val_loss=val_loss, val_accuracy=val_accuracy, learning_rate=optimizer.param_groups[0]["lr"],
                           best_val_accuracy=best_accuracy, best_epoch=best_epoch,
                           epochs_without_improvement=stale_epochs, is_best=int(improved),
                           epoch_seconds=time.perf_counter() - epoch_started, elapsed_seconds=time.perf_counter() - started)
                writer.writerow(row)
                handle.flush()
                logger.info("Epoch %03d/%03d | train loss=%.5f acc=%.4f | val loss=%.5f acc=%.4f | best=%.4f @%d | patience=%d/%d | %.1fs%s",
                            epoch, config.max_epochs, train_loss, train_accuracy, val_loss, val_accuracy,
                            best_accuracy, best_epoch, stale_epochs, config.patience, row["epoch_seconds"], " *" if improved else "")
                if stale_epochs >= config.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
        best = torch.load(fold_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
        result = {
            "status": "completed", "test_subject": subject, "validation_block": block + 1,
            "seed": fold_seed, "epochs_trained": epoch, "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy,
            "elapsed_seconds": time.perf_counter() - started, "fold_dir": str(fold_dir.resolve()),
        }
        write_json(fold_dir / "result.json", result)
        logger.info("Finished | best epoch=%d | test loss=%.5f acc=%.4f", best_epoch, test_loss, test_accuracy)
        return result
    except BaseException as error:
        logger.exception("Fold interrupted or failed; completed epoch artifacts remain in %s", fold_dir)
        write_json(fold_dir / "result.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                             "error": repr(error), "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        if history_path.exists():
            try:
                plot_history(history_path, fold_dir / "curves.png")
            except Exception:
                logger.exception("Could not render curves; history.csv remains available")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def summarize(results):
    rows = []
    for subject in sorted({row["test_subject"] for row in results}):
        values = [row["test_accuracy"] for row in results if row["test_subject"] == subject]
        rows.append({"test_subject": subject, "completed_folds": len(values),
                     "mean_test_accuracy": float(np.mean(values)), "std_test_accuracy": float(np.std(values))})
    return rows


def run_strict_loso(model_factory, config):
    """Run all requested fits. Each call creates a new, non-overwriting experiment."""
    config.validate()
    for subject in config.test_subjects:
        path = Path(config.data_dir) / str(subject) / f"bct_{subject}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir).resolve() / f"{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {
        "training": asdict(config), "python": sys.version, "torch": str(torch.__version__),
        "device": str(device), "model_factory": f"{model_factory.__module__}.{model_factory.__qualname__}",
        "selection": "Highest validation accuracy; earliest epoch wins ties; test evaluated only after training.",
        "aggregation": "Mean across inner validation folds, then mean across held-out test subjects; std uses ddof=0.",
    })
    results = []
    planned = len(config.test_subjects) * len(config.blocks)
    write_json(run_dir / "summary.json", {"status": "running", "planned_folds": planned, "completed_folds": 0})
    print(f"Run directory: {run_dir}\nPlanned fits: {len(config.test_subjects)} × {len(config.blocks)} = {planned}", flush=True)
    try:
        for subject in config.test_subjects:
            pool, test_data, source_path = load_subject_data(config, subject)
            for block in config.blocks:
                print(f"\nFit {len(results) + 1}/{planned}", flush=True)
                results.append(train_fold(model_factory, config, pool, test_data, subject, block, run_dir, device, source_path))
                write_csv(run_dir / "fold_results.csv", results)
                write_csv(run_dir / "subject_results.csv", summarize(results))
                write_json(run_dir / "summary.json", {"status": "running", "planned_folds": planned, "completed_folds": len(results)})
            del pool, test_data
    except BaseException as error:
        write_json(run_dir / "summary.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                              "planned_folds": planned, "completed_folds": len(results), "error": repr(error)})
        raise
    subjects = summarize(results)
    means = [row["mean_test_accuracy"] for row in subjects]
    summary = {"status": "completed", "planned_folds": planned, "completed_folds": len(results),
               "mean_subject_test_accuracy": float(np.mean(means)), "std_subject_test_accuracy": float(np.std(means)),
               "subjects": subjects}
    write_json(run_dir / "summary.json", summary)
    print(f"\nMean subject accuracy: {summary['mean_subject_test_accuracy']:.4f} ± {summary['std_subject_test_accuracy']:.4f}", flush=True)
    return run_dir, results
