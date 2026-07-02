# MIMIC Cardio-Oncology
- Author: Adrian Luis Balajadia
- Affiliation: Department of Biomedical Engineering at the University of Calgary
- Funding: Natural Sciences and Engineering Research Council of Canada (NSERC) Undergraduate Student Research Awards (USRA)


# Pipeline Usage Notes

End-to-end pipeline: **cohort → tokenization → summarize → train**

Splits are computed per training run from the run's seed — not generated during tokenization.  
This allows multi-seed experiments (seed = 42, 43, 44 …) each with independent patient assignments.

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

Add `--summarize` to print cohort statistics and save figures in the same command:

```bash
python tokenization_src/tokenize_cli.py \
    --data-dir <DATA_DIR> \
    --cohort cycle_modeling_ver2 \
    --name ver1 \
    --summarize
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | yes | — | Path to the MIMIC-IV raw data directory |
| `--cohort` | no | `cycle_modeling_ver2` | Source cohort directory under `cohort_outputs/` |
| `--name` | no | `ver1` | Output subdirectory under `tokenization_outputs/` |
| `--max-seq-len` | no | `600` | Maximum token sequence length (truncates oldest events) |
| `--summarize` | no | off | Print summary statistics and save figures after tokenizing |
| `--insert-att` | no | off | Insert CEHR-BERT Artificial Time Tokens (`W0`–`W3`, `M1`–`M11`, `LT`) between consecutive visits |
| `--insert-visit-delimiters` | no | off | Wrap each visit's events with `[V_START]`/`[V_END]` tokens |
| `--bucket-labs` | no | off | Append per-itemid quantile bucket (`_Q1`–`_Q4`) to abnormal lab tokens — changes vocab |
| `--bucket-medications` | no | off | Append per-drug dose-tier bucket (`_Q1`–`_Q4`) to medication tokens — changes vocab |

**Outputs** (`tokenization_outputs/<name>/`):

| File | Description |
|---|---|
| `concept_ids.pt` | Long tensor `(N, max_seq_len)` |
| `type_ids.pt` | Long tensor `(N, max_seq_len)` |
| `visit_ids.pt` | Long tensor `(N, max_seq_len)` |
| `position_ids.pt` | Long tensor `(N, max_seq_len)` |
| `dates.pt` | Long tensor `(N, max_seq_len)` — days since 2000-01-01 per token; used by time embedding |
| `age_ids.pt` | Long tensor `(N,)` — decade bucket (0–9); used in additive baseline |
| `age_years.pt` | Float tensor `(N,)` — continuous age in years; used by concat embedding |
| `labels.pt` | Long tensor `(N,)` — binary labels |
| `attention_mask.pt` | Bool tensor `(N, max_seq_len)` |
| `samples.csv` | Per-sample metadata (subject_id, cycle_number, prediction_time, binary_label, seq_len) |
| `samples.parquet` | Same as above in Parquet format |
| `vocab.json` | Concept and type vocabulary mappings (includes ATT tokens `W0`–`W3`, `M1`–`M11`, `LT`) |
| `metadata.json` | `max_seq_len`, `positive_rate`, vocab size, cohort stats, tokenisation flags |

You can also run the tokenizer module directly:

```bash
python tokenization_src/tokenize_cycle_sequences.py \
    --data-dir <DATA_DIR> \
    --cohort cycle_modeling_ver2 \
    --name ver1 \
    --max-seq-len 600 \
    --insert-att \
    --insert-visit-delimiters
```

### Sequence structure

Without optional flags:
```
[CLS] dx1 lab1 lab2 dx2 lab3 ...
```

With `--insert-visit-delimiters`:
```
[CLS] [V_START] dx1 lab1 [V_END] [V_START] dx2 lab3 [V_END] ...
```

With `--insert-att` and `--insert-visit-delimiters` (CEHR-BERT style):
```
[CLS] [V_START] dx1 lab1 [V_END] [W2] [V_START] dx2 [V_END] [LT] [V_START] lab3 [V_END]
         ^^^ visit 1 ^^^        ^ATT^      ^^^ visit 2 ^^^         ^ATT^   ^^^ visit 3 ^^^
```

ATT token thresholds (CEHR-BERT `CEHR_BERT` mode):

| Token | Inter-visit gap |
|---|---|
| `W0`–`W3` | 0–27 days (weekly bins) |
| `M1`–`M11` | 28–359 days (monthly bins) |
| `LT` | ≥ 360 days |

---

## Step 3 — Summarize (standalone)

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

## Step 4 — Train

Trains the BERT-style EHR encoder. The patient split (70 / 15 / 15) is computed at runtime from `samples.parquet` using `--seed`, so no separate split step is needed. The same seed also controls weight initialisation, making each run fully reproducible.

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
| `--data-dir` | `tokenization_outputs/ver1` | Tokenization directory |
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
| `--seed` | `42` | Controls both the patient split and weight initialisation |
| `--use-wandb` | off | Enable Weights & Biases experiment tracking |
| `--wandb-project` | `mimic-cardio-oncology` | W&B project name |
| `--run-name` | `None` | W&B run name (auto-generated if omitted) |
| `--fusion` | `add` | `add`: BEHRT-style element-wise sum. `concat`: CEHR-BERT concat→Linear→GELU |
| `--use-time` | off | Add sinusoidal time-gap embedding per token (requires `dates.pt`) |
| `--use-age` | off | Add continuous-age sinusoidal embedding (requires `age_years.pt`) |

**Outputs** (`experiment_outputs/<run>/`):

| File | Description |
|---|---|
| `best_model.pt` | State dict of the best checkpoint (by val AUROC) |
| `config.json` | All hyperparameters, derived vocab/seq sizes, hardware info, and run date |
| `history.json` | Per-epoch train loss, val loss, val AUROC, and elapsed time |
| `test_metrics.json` | AUROC and loss on the held-out test split, evaluated once after training |

---

## Model architecture

The model is a BERT-style encoder (`EHR_Encoder`) controlled by three orthogonal embedding flags set in `TrainConfig` or via CLI:

### Embedding flags

| Flag | Values | Effect |
|---|---|---|
| `fusion` | `"add"` (default) | BEHRT-style element-wise sum of all embedding tables |
| | `"concat"` | CEHR-BERT/EHRMamba: `cat([concept, time*, age*, position]) → Linear(4d→d) → GELU`, then type/visit/segment as additive residuals. Missing components (disabled by flags below) are zeroed before projection — keeping weight shape identical across B0–B2. |
| `use_time` | `False` (default) / `True` | Add learned sinusoidal time-gap per token: `sin((days/365.25) × w + φ)`. Requires `dates.pt`. |
| `use_age` | `False` (default) / `True` | Add continuous-age sinusoidal per token: `sin(age_years × w + φ)`. In `"add"` mode this is added on top of the discrete decade-bucket embedding already present. Requires `age_years.pt`. |

### Embedding components by configuration

| Component | `fusion="add"` | `fusion="concat"` |
|---|---|---|
| Concept | additive | in concat |
| Type | additive | additive residual |
| Visit | additive | additive residual |
| Segment (`visit_id % 2`) | additive | additive residual |
| Position | additive | in concat |
| Age (decade bucket) | additive, always | — |
| Time (`use_time=True`) | additive | in concat (zeroed if False) |
| Age sinusoidal (`use_age=True`) | additive | in concat (zeroed if False) |
| Projection | — | `nn.Linear(4d → d)` |

### Encoder

```
EHR_Event_Embedding → N × TransformerEncoderLayer (pre-norm, GELU FFN) → CLS pooling → Linear(d → 2)
```

- Optimizer: AdamW
- Scheduler: CosineAnnealingLR (`T_max=epochs`, `eta_min=lr/10`)
- Loss: CrossEntropyLoss with inverse-frequency class weights
- Mixed precision: `torch.amp.autocast` + `GradScaler` (CUDA only)
- Gradient clipping: `max_norm=1.0`
- Best checkpoint saved by validation AUROC

---

## Embedding ablations

### Ablation grid

| ID | `fusion` | `use_time` | `use_age` | Purpose |
|---|---|---|---|---|
| A0 | `"add"` | False | False | Baseline — no temporal info |
| A1 | `"add"` | True | False | Does time-gap signal help? |
| A2 | `"add"` | False | True | Does patient age help? |
| A3 | `"add"` | True | True | Best additive temporal version |
| B0 | `"concat"` | False | False | Tests concat fusion alone |
| B1 | `"concat"` | True | False | Concat + time |
| B2 | `"concat"` | True | True | CEHR-BERT/EHRMamba style |
| C1 | best of A/B | — | — | Same flags + `insert_att=True` tokenization |
| C2 | `"concat"` | True | True | Full CEHR-BERT (concat + time + age + ATT) |

### Tokenization requirements

`dates.pt` and `age_years.pt` are **always written** by the tokenizer. No special tokenization flag is needed for A0–B2.

C1/C2 require a separate tokenization built with `insert_att=True`. The `Jul1_512_att` variant is already defined in `run_tokenization.py`'s `RUNS` list — just run it:

```bash
python run_tokenization.py   # produces tokenization_outputs/Jul1_512_att
```

Then in `run_train.py`, uncomment the C-row in `ABLATIONS` and set its `data_dir` to `tokenization_outputs/Jul1_512_att`.

Tokenization-level flags (set in `run_tokenization.py` / `TokenizationConfig`):

| Flag | Effect |
|---|---|
| `insert_att=True` | ATT tokens between visits (`W0`–`W3`, `M1`–`M11`, `LT`) — needed for C1/C2 |
| `insert_visit_delimiters=True` | `[V_START]`/`[V_END]` around each hospital visit |
| `bucket_labs=True` | Append per-itemid quantile bucket (`_Q1`–`_Q4`) to abnormal lab tokens |
| `bucket_medications=True` | Append per-drug dose-tier bucket (`_Q1`–`_Q4`) to medication tokens |

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

Defines multiple tokenization variants in a `RUNS` list — same pattern as `run_train.py`. Each entry is a `TokenizationConfig` with its own `output_name` and flags, producing an independent folder under `tokenization_outputs/`.

```python
_BASE = dict(
    data_dir    = ...,
    cohort_name = "cycle_modeling_v3",
    max_seq_len = 512,
    run_summarize = True,
)

RUNS = [
    TokenizationConfig(**_BASE, output_name="Jul1_512"),                          # base
    TokenizationConfig(**_BASE, output_name="Jul1_512_att",   insert_att=True),   # C1/C2
    TokenizationConfig(**_BASE, output_name="Jul1_512_bucketed_labs",  bucket_labs=True),
    TokenizationConfig(**_BASE, output_name="Jul1_512_bucketed_meds",  bucket_medications=True),
    TokenizationConfig(**_BASE, output_name="Jul1_512_bucketed_all",   bucket_labs=True, bucket_medications=True),
]
```

Comment out any variants you don't need, then:

```bash
python run_tokenization.py
```

Each run prints a header summarising its active flags before tokenizing.

### `run_train.py`

The file you edit most often. Define ablations in the `ABLATIONS` list and seeds in `SEEDS` — the script generates and runs every combination sequentially.

Each `seed` value independently controls two things:
- **Patient split** — which patients go to train / val / test (70 / 15 / 15, computed at runtime)
- **Model initialisation** — weight init and dropout randomness

**Ablation + multi-seed pattern** (current default):

```python
ABLATIONS = [
    ("A0", dict(fusion="add",    use_time=False, use_age=False)),
    ("A1", dict(fusion="add",    use_time=True,  use_age=False)),
    ("A2", dict(fusion="add",    use_time=False, use_age=True)),
    ("A3", dict(fusion="add",    use_time=True,  use_age=True)),
    ("B0", dict(fusion="concat", use_time=False, use_age=False)),
    ("B1", dict(fusion="concat", use_time=True,  use_age=False)),
    ("B2", dict(fusion="concat", use_time=True,  use_age=True)),
]

SEEDS = [42, 43, 44, 45, 46]

RUNS = [
    TrainConfig(**_BASE, **emb_kwargs, seed=s,
                output_dir=Path(f"experiment_outputs/Jul1_ablations/{ablation_id}/seed{s}"),
                run_name=f"{ablation_id}-seed{s}")
    for ablation_id, emb_kwargs in ABLATIONS
    for s in SEEDS
]
```

This produces 35 runs (7 ablations × 5 seeds). Average `test_metrics.json` across seeds to report mean ± std AUROC per ablation.

```bash
python run_train.py
```

Each run saves its own `config.json`, `history.json`, `best_model.pt`, and `test_metrics.json` to `output_dir`.  
Set `use_wandb = True` in `_BASE` to enable Weights & Biases logging for all runs.

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

Runs a trained checkpoint on a full data split and reports aggregate metrics and a per-sample result table. The split is reconstructed from the seed in `config.json`, so it exactly matches the split used during training — no `splits.json` needed.

Note: `train.py` already evaluates the test split automatically and saves `test_metrics.json`. Use this script for deeper per-sample analysis or to re-evaluate on a different split.

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
| `--split` | `test` | Which split to evaluate on (`train`, `val`, `test`) |
| `--batch-size` | `32` | Inference batch size |
| `--threshold` | `0.5` | Decision threshold for binary predictions |
| `--max-rows` | `50` | Max rows shown in the per-sample table |
| `--output-csv` | — | Optional path to save full per-sample results |

### `evaluation/compare_ablations.py`

Scans an experiment directory for all `test_metrics.json` files, groups them by ablation ID, and reports mean ± std AUROC across seeds. Also produces a bar chart with error bars.

```bash
# Print summary table
python evaluation/compare_ablations.py experiment_outputs/Jul1_ablations/

# Save bar chart
python evaluation/compare_ablations.py experiment_outputs/Jul1_ablations/ \
    --save experiment_outputs/Jul1_ablations/comparison.png

# Sort by mean AUROC instead of ablation name
python evaluation/compare_ablations.py experiment_outputs/Jul1_ablations/ --sort auroc

# Table only, no plot
python evaluation/compare_ablations.py experiment_outputs/Jul1_ablations/ --no-plot
```

| Argument | Default | Description |
|---|---|---|
| `root` | required | Root experiment directory containing `<ablation>/<seed>/test_metrics.json` |
| `--save` | — | Save bar chart to this path (PNG/PDF). Omit to display interactively |
| `--sort` | `id` | Sort rows by `id` (ablation name) or `auroc` (best first) |
| `--no-plot` | off | Print table only, skip the bar chart |
| `--dpi` | `150` | Output DPI when saving |

Expected directory layout:
```
experiment_outputs/Jul1_ablations/
  A0/seed42/test_metrics.json
  A0/seed43/test_metrics.json
  A1/seed42/test_metrics.json
  ...
```

---

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

# 2. Tokenize and summarize (no split — computed per training run)
python tokenization_src/tokenize_cli.py \
    --data-dir <DATA_DIR> \
    --cohort cycle_modeling_ver2 \
    --name ver1 \
    --summarize

# 3. Train (split computed from --seed; test metrics saved to experiment_outputs/run1/test_metrics.json)
python model_src/train.py \
    --data-dir tokenization_outputs/ver1 \
    --output-dir experiment_outputs/run1 \
    --seed 42
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

### Running the full embedding ablation sweep (A0–B2, 5 seeds)

A0–B2 all use the same tokenization — `dates.pt` and `age_years.pt` are always written.  
No re-tokenization needed.

```bash
# Edit _BASE["data_dir"] in run_train.py to point at your tokenization, then:
python run_train.py
# Runs 35 jobs: 7 ablations × 5 seeds
# Results: experiment_outputs/Jul1_ablations/<ID>/seed<N>/test_metrics.json
```

To run a single ablation manually (e.g. A1, seed 42):

```bash
python model_src/train.py \
    --data-dir tokenization_outputs/Jun30_512 \
    --output-dir experiment_outputs/test_A1 \
    --fusion add --use-time \
    --seed 42 --epochs 100
```

### Running C1/C2 (requires ATT tokenization)

```bash
# 1. Build ATT tokenization (Jul1_512_att is already in run_tokenization.py's RUNS list)
python run_tokenization.py

# 2. In run_train.py: uncomment the C2 row in ABLATIONS and set data_dir="tokenization_outputs/Jul1_512_att"
python run_train.py
```

### Multi-seed experiment (variance estimation)

```bash
# After tokenizing once, train with seeds 42–46
for SEED in 42 43 44 45 46; do
    python model_src/train.py \
        --data-dir tokenization_outputs/ver1 \
        --output-dir experiment_outputs/A0_seed${SEED} \
        --fusion add \
        --seed ${SEED}
done
# Average test_metrics.json across seeds for mean ± std AUROC
```

Or define all seeds in the `RUNS` list in `run_train.py`.
