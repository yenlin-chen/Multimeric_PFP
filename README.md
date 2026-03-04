# Multimeric_PFP

Graph neural network workflows for protein function prediction (PFP) on monomeric and multimeric protein assemblies, using sequence embeddings and ANM-derived dynamical couplings.

## Overview

This repository contains:

- **Data curation scripts** for building a homo-multimer dataset from public protein resources.
- **Core Python modules** for graph construction, embeddings, model architecture, training, and metrics.
- **Experiment workspaces** with train/valid/test splits and sweep scripts.

The pipeline builds residue-level graphs from protein structures, combines edge types (e.g., contact/codir/coord/deform), and trains multi-label classifiers for GO-term prediction.

## Repository Layout

- `src/modules/data/`
  - `datasets.py`: creates/loads graph datasets and handles preprocessing hooks.
  - `encoders.py`: sequence encoders (`ProtTrans`, `ProteinBERT`) via Hugging Face.
  - `enm.py`: ANM-based coupling computation (via ProDy).
  - `retrievers.py`: structure retrieval utilities.
- `src/modules/training/`
  - `model_arch.py`: `SimplifiedMultiGCN` model definition.
  - `trainer.py`: training/evaluation loop wrapper.
  - `metrics.py`: classification/regression metrics.
  - `visualization.py`: plotting/logging utilities.
- `src/data_curation/`
  - notebooks and scripts used to construct the curated dataset.
- `data/`
  - `external/`: raw downloaded assets (e.g., PDB).
  - `collated/`: intermediate generated features (e.g., ANM outputs).
  - `processed/`: graph-ready data artifacts.
- `experiments/`
  - DR1 (`20250909 ...`) and DR2 (`20251015 ...`) split files and sweep scripts.

## Requirements

No lockfile/environment file is currently included. Create a Python environment (3.10+ recommended), then install the required dependencies used by `src/modules` and experiment scripts.

Use the repository requirements file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Core packages

- `numpy`
- `torch`
- `torch-geometric`
- `torchinfo`
- `prody`
- `transformers`
- `tqdm`
- `requests`
- `matplotlib`
- `wandb`

> Notes:
> - `torch`/`torch-geometric` installation can be CUDA-version specific. Use the official install selectors for your system.
> - First use of `ProtTrans` / `ProteinBERT` will download large pretrained checkpoints.

## Data Expectations

The code expects this root-level structure:

- `data/external/Protein_Data_Bank/pdb-biomt/*.pdb`
- `data/collated/ANM/...` (generated during preprocessing)
- `data/processed/...` (generated graph/embedding artifacts)

Experiment scripts also expect split files in each experiment root, e.g.:

- `train_identifiers.txt`, `valid_identifiers.txt`, `test_identifiers.txt`
- `train_annotations.csv`, `valid_annotations.csv`, `test_annotations.csv`

These files are already present under DR1/DR2 in this repository.

## Running Experiments

Most training entrypoints are sweep scripts under `experiments/.../sweep.py` and use Weights & Biases sweeps.

### Example: DR2 multimeric full sweep

```bash
cd "experiments/20251015 full dataset test (DR2)/multimeric/full_sweep"
python sweep.py
```

### Example: DR2 monomeric contact full sweep

```bash
cd "experiments/20251015 full dataset test (DR2)/monomeric/full_sweep/contact"
python sweep.py
```

### Example: DR1 multimeric basic sweep (contact)

```bash
cd "experiments/20250909 full dataset test (DR1)/multimeric/basic_sweep/contact"
python sweep.py
```

Each sweep script defines hyperparameter search spaces and starts a W&B agent in `__main__`.

## Module Usage (Programmatic)

From repository root:

```python
import numpy as np
from src.modules.data.datasets import Dataset

assembly_ids = np.loadtxt("experiments/20251015 full dataset test (DR2)/train_identifiers.txt", dtype=np.str_)
annotations = np.loadtxt("experiments/20251015 full dataset test (DR2)/train_annotations.csv", delimiter=",", dtype=np.int32)

dataset = Dataset(
    pdb_assembly_ids=assembly_ids,
    annotations=annotations,
    version="v1",
    sequence_embedding="ProtTrans",
    enm_type="anm",
    use_monomers=False,
    thresholds={"contact": "12", "codir": "1CONT", "coord": "1CONT", "deform": "1CONT"},
)
```

## Reproducibility Notes

- Sweep scripts set a fixed random seed (`69`) for PyTorch/NumPy worker setup.
- Model checkpoints such as `model-best_loss.pt` and `model-best_f1_max.pt` are saved in each run directory.

## License

This project is licensed under the terms in `LICENSE`.
