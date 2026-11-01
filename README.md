# NeuroMambaNet

PyTorch implementation and pretrained checkpoints for **NeuroMambaNet: A Prior-Integrated Dual-Stream Mamba Network for Robust Cross-Subject Motor Imagery EEG Classification**.

NeuroMambaNet combines a DWT-based frequency stream with a raw EEG stream. Mamba blocks model multiple frequency bands, while brain-region grouping and a beta-rhythm gate combine complementary spatial features for cross-subject EEG classification.

## Repository structure

```text
NeuroMambaNet/
├── model/
│   ├── NeuroMambaNet.py
│   ├── mambo.py
│   └── DWT.py
├── checkpoints/
│   ├── BCI_2a/
│   └── CBCIC/
├── strict_loso.py
├── validation.py
├── predict.ipynb
└── README.md
```

## Datasets

| Dataset | Description | Access |
| --- | --- | --- |
| BCI Competition IV Dataset 2a | Four-class motor imagery: left hand, right hand, feet and tongue; 9 participants, 22 EEG channels, 250 Hz. | [Official download page](https://www.bbci.de/competition/iv/) · [Dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf) |
| Clinical Brain–Computer Interface Challenge 2020 (CBCIC) | Left/right-hand motor attempts in stroke patients; 12 EEG channels, 512 Hz. | [Dataset and documentation](https://github.com/5anirban9/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow) |

For BCI IV-2a, select **Data sets 2a** in the download section of the official website. For CBCIC, this study uses the labeled calibration (`T`) files from **P01–P08**; the unlabeled evaluation (`E`) files are not used. Raw datasets are downloaded separately from the links above.

## Dependencies

- [PyTorch](https://pytorch.org/)
- [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets)
- [mamba-ssm](https://github.com/state-spaces/mamba), including `Mamba2`

Follow the Mamba installation instructions for a compatible PyTorch and CUDA environment.


## Checkpoint evaluation

Open `predict.ipynb` from the project root and run the cells in order. Import the evaluation helper with `from validation import evaluate_dataset`. Set `DATASET` to `"BCI2A"` or `"CBCIC"` to load the corresponding checkpoints and display subject-level accuracy, Cohen's kappa, summary statistics and the normalized confusion matrix. Results are displayed directly in the notebook without exporting result files. Dependencies are `torch`, `numpy`, `pandas`, `scikit-learn` and `pytorch-wavelets`.

The notebook expects these preprocessed local files, prepared separately from the raw dataset downloads:

- **BCI IV-2a:** `data/BCI2A/{subject}/bct_{subject}.pkl`, subjects 1–9. Each file contains `(pool_X, pool_y, test_X, test_y)`; only the test pair is evaluated.
- **CBCIC:** `data/CBCIC/cbcic_data.pkl`, containing `[X, y]` for P01–P08 in subject order, with 80 samples per subject.

Each dataset run evaluates all 72 BCI IV-2a or 56 CBCIC checkpoints. For a single-fold check, call `evaluate_dataset("BCI2A", subjects=[1], folds=[0])`. Data preprocessing is not repeated during evaluation.

## Strict LOSO training

`strict_loso.py` provides `TrainConfig` and `run_strict_loso(model_factory, config)` for training with one held-out test subject and an inner validation block. For the local BCI IV-2a files, set `TrainConfig(data_dir="data/BCI2A")` and supply a model factory. Checkpoints are selected using validation accuracy; the held-out test subject is evaluated after training. Each run saves checkpoints, logs and training summaries in its own output folder. Training-curve export uses `matplotlib`.
