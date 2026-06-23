# MIMIC Cardio-Oncology
- Author: Adrian Luis Balajadia
- Affiliation: Department of Biomedical Engineering at the University of Calgary
- Funding: Natural Sciences and Engineering Research Council of Canada (NSERC) Undergraduate Student Research Awards (USRA)

This repo


# Pipeline Usage Notes

End-to-end pipeline: **cohort → tokenization → split → summarize → train**

All scripts are run from the **repo root** unless noted otherwise.

---

## Two ways to run the pipeline

| | CLI | Runner scripts |
|---|---|---|
| **How** | `python cohort_src/cohort_cli.py --data-dir ...` | `python run_cohort.py` |
| **Config lives in** | Command-line arguments | `run_*.py` files, version-controlled |
| **Best for** | One-off runs, shell scripts, HPC job submission | Development, experiments, ablations, notebooks |
| **Multi-run** | Shell loop or separate invocations | Add entries to `RUNS` list in `run_train.py` |

Both approaches call the same underlying `main()` functions — they are interchangeable and complementary, not competing.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

You will need access to the raw MIMIC-IV data directory, referred to below as `<DATA_DIR>`.  
Expected layout:

```
<DATA_DIR>/
  mimic-iv-3.1/hosp/      ← prescriptions.csv, diagnoses_icd.csv, …
  mimic-iv-echo/           ← echo report CSVs
```

---

## Step 1 — Build the cohort

Runs a chain of DuckDB SQL files to produce the cycle-level cardiotoxicity modelling table.

```bash
python cohort_src/cohort_cli.py \
    --data-dir <DATA_DIR> \
    --name cycle_modeling_ver2
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | yes | — | Path to the MIMIC-IV raw data directory |
| `--name` | no | `cycle_modeling_ver2` | Output subdirectory under `cohort_outputs/` |

**Outputs** (`cohort_outputs/<name>/`):

| File | Description |
|---|---|
| `final_cycle_modeling_table.csv/.parquet` | One row per (patient, cycle), multi-class label |
| `final_cycle_binary_modeling_table.csv/.parquet` | Same rows, binary label collapsed |
| `row_level_label_breakdown.csv` | Cycle counts per label |
| `row_level_binary_label_breakdown.csv` | Cycle counts per binary label |
| `row_level_drug_class_breakdown.csv` | Cycle counts per drug class combination |
| `patient_level_labels.csv` | One row per patient with assigned status |
| `patient_level_summary.csv` | Patient counts per status |
| `cohort_accounting.csv` | High-level patient counts at each pipeline stage |

You can also run the module directly:

```bash
python cohort_src/generate_cycle_modeling_table.py \
    --data-dir <DATA_DIR> \
    --name cycle_modeling_ver2
```

---

## Step 2 — Tokenize

Converts the cohort CSV into padded integer token tensors ready for PyTorch.

```bash
python tokenization_src/tokenize_cli.py \
    --data-dir <DATA_DIR> \
    --cohort cycle_modeling_ver2 \
    --name ver1
```

Add `--split` and/or `--summarize` (or `--all`) to run those steps in one command:

```bash
# Tokenize + split + summarize in one go
python tokenization_src/tokenize_cli.py \
    --data-dir <DATA_DIR> \
    --cohort cycle_modeling_ver2 \
    --name ver1 \
    --all
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | yes | — | Path to the MIMIC-IV raw data directory |
| `--cohort` | no | `cycle_modeling_ver2` | Source cohort directory under `cohort_outputs/` |
| `--name` | no | `ver1` | Output subdirectory under `tokenization_outputs/` |
| `--max-seq-len` | no | `600` | Maximum token sequence length (truncates oldest events) |
| `--split` | no | off | Run stratified patient-level split after tokenizing |
| `--summarize` | no | off | Print summary statistics and save figures after tokenizing |
| `--all` | no | off | Equivalent to `--split --summarize` |

**Outputs** (`tokenization_outputs/<name>/`):

| File | Description |
|---|---|
| `concept_ids.pt` | Long tensor `(N, max_seq_len)` |
| `type_ids.pt` | Long tensor `(N, max_seq_len)` |
| `visit_ids.pt` | Long tensor `(N, max_seq_len)` |
| `position_ids.pt` | Long tensor `(N, max_seq_len)` |
| `age_ids.pt` | Long tensor `(N,)` — one per sample |
| `labels.pt` | Long tensor `(N,)` — binary labels |
| `samples.parquet` | Metadata table (subject_id, cycle_number, binary_label, …) |
| `vocab.json` | Concept and type vocabulary mappings |
| `metadata.json` | `max_seq_len`, `positive_rate`, cohort name |

You can also run the tokenizer module directly:

```bash
python tokenization_src/tokenize_cycle_sequences.py \
    --data-dir <DATA_DIR> \
    --cohort cycle_modeling_ver2 \
    --name ver1 \
    --max-seq-len 600
```

---

## Step 3 — Split (standalone)

Creates a stratified patient-level train / val / test split (70 / 15 / 15).  
Splitting is at the **patient level** — all cycles for a patient go to the same partition.

```bash
python tokenization_src/split_dataset.py tokenization_outputs/ver1
```

Omit the path argument to use the default (`tokenization_outputs/ver1`):

```bash
python tokenization_src/split_dataset.py
```

**Outputs** (written into the same directory):

| File | Description |
|---|---|
| `splits.json` | Subject ID lists and row indices for train / val / test |
| `splits_summary.csv` | Per-split patient count, sample count, positive rate |

---

## Step 4 — Summarize (standalone)

Prints cohort statistics and saves matplotlib figures.  
Can be re-run at any time without re-tokenizing.

```bash
python tokenization_src/summarize_tokenization.py tokenization_outputs/ver1
```

Omit the path argument to use the default (`tokenization_outputs/ver1`).

**Figures saved** (`tokenization_outputs/<name>/figures/`):

| File | Description |
|---|---|
| `label_distribution.png` | Positive / negative sample counts |
| `sequence_length_histogram.png` | Distribution of sequence lengths with truncation line |
| `vocabulary_breakdown.png` | Token counts by event type |
| `age_distribution.png` | Distribution of patient age decade buckets |
| `split_summary.png` | Train / val / test patient and sample counts (requires `splits.json`) |

---

## Step 5 — Train

Trains the BERT-style EHR encoder on the tokenized and split dataset.  
Requires `splits.json` to exist in the data directory (run Step 3 first).

```bash
python model_src/train.py \
    --data-dir tokenization_outputs/ver1 \
    --output-dir model_outputs/run1
```

Quick debug run with a smaller model:

```bash
python model_src/train.py \
    --data-dir tokenization_outputs/ver1 \
    --output-dir model_outputs/debug \
    --d-model 64 --num-heads 4 --num-layers 2 \
    --epochs 3 --batch-size 16
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `tokenization_outputs/ver1` | Tokenization directory (must contain `splits.json`) |
| `--output-dir` | `model_outputs/run1` | Where to save checkpoints and logs |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight-decay` | `1e-2` | AdamW weight decay |
| `--d-model` | `128` | Embedding / hidden dimension |
| `--num-heads` | `4` | Attention heads (must divide `d-model`) |
| `--num-layers` | `4` | Number of TransformerEncoder layers |
| `--ff-dim` | `512` | Feed-forward inner dimension |
| `--dropout` | `0.1` | Dropout probability |
| `--num-workers` | `0` | DataLoader worker processes |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |

**Outputs** (`model_outputs/<run>/`):

| File | Description |
|---|---|
| `best_model.pt` | State dict of the best checkpoint (by val AUROC) |
| `config.json` | All hyperparameters + derived vocab/seq sizes |
| `history.json` | Per-epoch train loss, val loss, val AUROC |

---

## Runner scripts (developer workflow)

Config dataclasses live in `configs/` and are imported by the runner scripts at the repo root.
Edit the config at the top of each runner, then execute it — no CLI flags needed.

### `run_cohort.py`

Edit `data_dir` and `output_name`, then:

```bash
python run_cohort.py
```

### `run_tokenization.py`

Edit `data_dir`, `cohort_name`, `output_name`, `max_seq_len`, and the `run_split` / `run_summarize` flags, then:

```bash
python run_tokenization.py
```

### `run_train.py`

The file you edit most often. Define one or more `TrainConfig` objects in the `RUNS` list — each gets its own `output_dir`. The script iterates through all of them sequentially.

```python
# run_train.py — example with two configs
RUNS = [
    TrainConfig(
        data_dir   = Path("tokenization_outputs/ver1"),
        output_dir = Path("model_outputs/baseline"),
        d_model    = 128,
        num_layers = 4,
    ),
    TrainConfig(
        data_dir   = Path("tokenization_outputs/ver1"),
        output_dir = Path("model_outputs/deeper"),
        d_model    = 256,
        num_layers = 8,
        ff_dim     = 1024,
    ),
]
```

```bash
python run_train.py
```

Each run saves its own `config.json` to `output_dir` alongside the checkpoint.

### `run_pipeline.py`

Full end-to-end run. Reads configs from the three stage scripts above — no duplication.
Use this for reproducibility or first-time setup.

```bash
python run_pipeline.py
```

Toggle stages with the `RUN_*` flags at the top of the file.

### Config serialization

Configs can be saved and reloaded as JSON for experiment tracking:

```python
cfg.save("model_outputs/run1/config.json")
cfg = TrainConfig.load("model_outputs/run1/config.json")
```

`run_train.py` does this automatically for every run.

---

## Smoke tests

Verify individual modules without the full dataset:

```bash
# Embedding layer
python model_src/embedding_layers.py

# Encoder architecture
python model_src/ehr_encoder.py

# Dataset / DataLoader (requires tokenization_outputs/ver1)
python model_src/dataset.py tokenization_outputs/ver1
```

---

## Typical full run

**Via CLI:**

```bash
# 1. Build cohort
python cohort_src/cohort_cli.py \
    --data-dir <DATA_DIR> \
    --name cycle_modeling_ver2

# 2. Tokenize, split, and summarize
python tokenization_src/tokenize_cli.py \
    --data-dir <DATA_DIR> \
    --cohort cycle_modeling_ver2 \
    --name ver1 \
    --all

# 3. Train
python model_src/train.py \
    --data-dir tokenization_outputs/ver1 \
    --output-dir model_outputs/run1
```

**Via runner scripts** (edit `data_dir` in each file first):

```bash
python run_cohort.py
python run_tokenization.py
python run_train.py
```

Or all at once:

```bash
python run_pipeline.py
```
