# MIMIC Cardio-Oncology
- Author: Adrian Luis Balajadia
- Affiliation: Department of Biomedical Engineering at the University of Calgary
- Funding: Natural Sciences and Engineering Research Council of Canada (NSERC) Undergraduate Student Research Awards (USRA)


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
| `samples.csv` | Per-sample metadata (subject_id, cycle_number, prediction_time, binary_label, seq_len) |
| `samples.parquet` | Same as above in Parquet format |
| `vocab.json` | Concept and type vocabulary mappings |
| `metadata.json` | `max_seq_len`, `positive_rate`, vocab size, cohort stats |

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

**Figures saved** (`tokenization_outputs/<name>/summarization_figures/`):

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
    --output-dir experiment_outputs/run1
```

Quick debug run with a smaller model:

```bash
python model_src/train.py \
    --data-dir tokenization_outputs/ver1 \
    --output-dir experiment_outputs/debug \
    --d-model 64 --num-heads 4 --num-layers 2 \
    --epochs 3 --batch-size 16
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `tokenization_outputs/ver1` | Tokenization directory (must contain `splits.json`) |
| `--output-dir` | `experiment_outputs/run1` | Where to save checkpoints and logs |
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
| `--seed` | `42` | Random seed for reproducibility |
| `--use-wandb` | off | Enable Weights & Biases experiment tracking |
| `--wandb-project` | `mimic-cardio-oncology` | W&B project name |
| `--run-name` | `None` | W&B run name (auto-generated if omitted) |

**Outputs** (`experiment_outputs/<run>/`):

| File | Description |
|---|---|
| `best_model.pt` | State dict of the best checkpoint (by val AUROC) |
| `config.json` | All hyperparameters, derived vocab/seq sizes, hardware info, and run date |
| `history.json` | Per-epoch train loss, val loss, val AUROC, and elapsed time |

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
        output_dir = Path("experiment_outputs/baseline"),
        d_model    = 128,
        num_layers = 4,
        seed       = 42,
    ),
    TrainConfig(
        data_dir   = Path("tokenization_outputs/ver1"),
        output_dir = Path("experiment_outputs/deeper"),
        d_model    = 256,
        num_layers = 8,
        ff_dim     = 1024,
        use_wandb  = True,       # enable W&B for this run
        run_name   = "deeper-ablation",
    ),
]
```

```bash
python run_train.py
```

Each run saves its own `config.json` to `output_dir` alongside the checkpoint.  
Set `use_wandb = True` on any config to enable Weights & Biases logging for that run.

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
cfg.save("experiment_outputs/run1/config.json")
cfg = TrainConfig.load("experiment_outputs/run1/config.json")
```

`run_train.py` does this automatically for every run.

---

## Data exploration

Scripts and notebooks for inspecting the tokenized dataset live in `data_exploration/`.

### `data_exploration/inspect_patient.py`

Visualize a single patient's tokenized EHR sequence in the terminal, and optionally run a model prediction on it.

```bash
# Random patient from the test split
python data_exploration/inspect_patient.py

# Specific index within the split (0-based)
python data_exploration/inspect_patient.py --patient-idx 5

# Look up by MIMIC subject_id directly
python data_exploration/inspect_patient.py --subject-id 13595646

# Subject with multiple chemotherapy cycles — pick cycle 1 (0-based)
python data_exploration/inspect_patient.py --subject-id 13595646 --cycle-idx 1

# Attach a model prediction
python data_exploration/inspect_patient.py --patient-idx 5 --model-dir experiment_outputs/run1

# Show all events per visit instead of truncating
python data_exploration/inspect_patient.py --patient-idx 5 --max-per-visit 0
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `tokenization_outputs/ver1` | Tokenization directory |
| `--split` | `test` | Which split to sample from (`train`, `val`, `test`) |
| `--patient-idx` | random | 0-based index within the split |
| `--subject-id` | — | MIMIC subject_id (searches across all splits) |
| `--cycle-idx` | `0` | Which cycle to show when a subject has multiple |
| `--model-dir` | — | Experiment dir with `config.json` + `best_model.pt` |
| `--max-per-visit` | `20` | Max events shown per visit (`0` = all) |

Requires `rich`:  `pip install rich`

---

## Evaluation

Post-training evaluation scripts live in `evaluation/`.

### `evaluation/evaluate_model.py`

Runs a trained checkpoint on a full data split and reports aggregate metrics and a per-sample result table.

```bash
# Evaluate on test split (default)
python evaluation/evaluate_model.py --model-dir experiment_outputs/run1

# Evaluate on val split
python evaluation/evaluate_model.py --model-dir experiment_outputs/run1 --split val

# Save per-sample predictions to CSV
python evaluation/evaluate_model.py --model-dir experiment_outputs/run1 \
    --output-csv experiment_outputs/run1/test_results.csv
```

Reports: AUROC, accuracy, precision, recall, F1, confusion matrix, and a per-sample table with subject_id, cycle, true label, predicted label, and P(cardiotoxic).

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | required | Experiment dir with `config.json` + `best_model.pt` |
| `--data-dir` | from config | Tokenization directory (read from `config.json` if omitted) |
| `--split` | `test` | Which split to evaluate on |
| `--batch-size` | `32` | Inference batch size |
| `--threshold` | `0.5` | Decision threshold for binary predictions |
| `--max-rows` | `50` | Max rows shown in the per-sample table |
| `--output-csv` | — | Optional path to save full per-sample results |

### `evaluation/plot_history.py`

Plots training loss and validation AUROC curves from `history.json`. Supports comparing multiple runs side by side.

```bash
# Single run — display interactively
python evaluation/plot_history.py --model-dir experiment_outputs/run1

# Save to PNG
python evaluation/plot_history.py --model-dir experiment_outputs/run1 \
    --save experiment_outputs/run1/training_curves.png

# Compare multiple runs on the same plot
python evaluation/plot_history.py \
    --model-dir experiment_outputs/run1 experiment_outputs/run2 \
    --save comparison.png
```

Requires `matplotlib`: `pip install matplotlib`

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
    --output-dir experiment_outputs/run1
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
