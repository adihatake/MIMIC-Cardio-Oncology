# MIMIC Cardio-Oncology
- Author: Adrian Luis Balajadia
- Affiliation: Department of Biomedical Engineering at the University of Calgary
- Funding: Natural Sciences and Engineering Research Council of Canada (NSERC) Undergraduate Student Research Awards (USRA)


# Pipeline Usage Notes

End-to-end pipeline: **cohort → tokenization → train → interpret**

Splits are computed per training run from the run's seed — not generated during tokenization.  
This allows multi-seed experiments (seed = 42, 52, 62, 72, 82) each with independent patient assignments.

All scripts are run from the **repo root** unless noted otherwise.

---

## Two ways to run the pipeline

| | Runner scripts | CLI |
|---|---|---|
| **How** | `python run_cohort.py` | `python cohort_src/cohort_cli.py --data-dir ...` |
| **Config lives in** | `run_*.py` files, version-controlled | Command-line arguments |
| **Best for** | Development, experiments, ablations | One-off runs, shell scripts, HPC job submission |
| **Multi-run** | Add entries to `RUNS` list in the relevant runner | Shell loop or separate invocations |

Both approaches call the same underlying `main()` functions — they are interchangeable.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Integrated Gradients (optional, Step 5) requires Captum:

```bash
pip install captum
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

### Runner script

Edit `OUTPUT_NAME`, `CYCLE_SQL_DIR`, and `PRESCRIPTIONS_SQL_DIR` at the top of `run_cohort.py`, then:

```bash
python run_cohort.py
```

**Pipeline options** — set `PIPELINE` in `run_cohort.py`:

| `PIPELINE` | Description |
|---|---|
| `"pan_cancer_ctrcd"` | 10 pan-cancer drug classes, combined CTRCD endpoint, uniform 365-day window |
| `"pan_cancer_bimodal"` | Same as `pan_cancer_ctrcd` but acute (90d) vs moderate (365d) windows |
| `"broad_cv"` | Same as `pan_cancer_ctrcd` but pre-existing arrhythmia also excluded from strict table |
| `"narrow_cv"` | Same as `pan_cancer_ctrcd` but arrhythmia/conduction codes excluded from endpoint |
| `"hf_cardiotox_v2"` | Combined CTRCD endpoint (ICD HF + LVEF + GLS), 3 drug classes, ESC 2022 windows |
| `"anthracycline_only_exposure"` | Anthracycline-first patients only, combined CTRCD endpoint |

Each pipeline automatically resolves its SQL directories from `sql_files/drug_cycles_sql/<pipeline>/` and `sql_files/prescriptions_sql/<pipeline>/`. Override with `CYCLE_SQL_DIR` / `PRESCRIPTIONS_SQL_DIR` only when pointing at a non-default subdirectory.

**Canonical table convention:** `final_cycle_binary_modeling_table.parquet` is always the **inclusive** table (retains patients with pre-existing HF/CMP — largest N for prediction models). The strict table is written separately as `ctrcd_final_binary_modeling_table_strict.parquet`.

Example `run_cohort.py` configuration:

```python
PIPELINE    = "pan_cancer_ctrcd"
OUTPUT_NAME = "pan_cancer_ctrcd_v3"
```

### CLI

```bash
python cohort_src/cohort_cli.py \
    --data-dir <DATA_DIR> \
    --pipeline pan_cancer_ctrcd \
    --name pan_cancer_ctrcd_v3
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | yes | — | Path to the MIMIC-IV raw data directory |
| `--pipeline` | no | `pan_cancer_ctrcd` | Pipeline name (see options above) |
| `--name` | no | `cycle_modeling_ver2` | Output subdirectory under `cohort_outputs/` |

**Cohort definition and label assignment** (`sql_files/drug_cycles_sql/<pipeline>/`):

Each (patient, chemotherapy cycle) row receives one of four labels:

| Label | Meaning |
|---|---|
| `positive` | A cardiotoxicity event occurred within the prediction window |
| `negative_observed` | No event, and the patient was observed through the end of the window |
| `exclude_already_toxic` | A cardiotoxicity event predated the prediction time |
| `unknown_no_followup_evidence` | No event and no follow-up evidence reaching the window |

`final_cycle_binary_modeling_table` retains only `positive` and `negative_observed` rows — unknown and already-toxic cases are excluded before any modelling step.

**Outputs** (`cohort_outputs/<name>/`):

| File | Description |
|---|---|
| `final_cycle_modeling_table.csv/.parquet` | One row per (patient, cycle), multi-class label |
| `final_cycle_binary_modeling_table.csv/.parquet` | Filtered to `positive` + `negative_observed` rows only |
| `row_level_label_breakdown.csv` | Cycle counts per label |
| `row_level_binary_label_breakdown.csv` | Cycle counts per binary label |
| `row_level_drug_class_breakdown.csv` | Cycle counts per drug class combination |
| `patient_level_labels.csv` | One row per patient with assigned status |
| `patient_level_summary.csv` | Patient counts per status |
| `cohort_accounting.csv` | High-level patient counts at each pipeline stage |

---

## Step 2 — Tokenize

Converts the cohort CSV into padded integer token tensors ready for PyTorch.

### Runner script

Edit `cohort_name` and add `TokenizationConfig` entries to `RUNS` in `run_tokenization.py`, then:

```bash
python run_tokenization.py
```

Example configuration for pan_cancer_ctrcd:

```python
_BASE = dict(
    cohort_name             = "pan_cancer_ctrcd_v3",
    max_seq_len             = 512,
    insert_visit_delimiters = True,
    include_all_labs        = True,
    ...
)
RUNS = [
    TokenizationConfig(**_BASE, output_name="pan_cancer_ctrcd_v3_all_labs",     cardiac_labs_only=False),
    TokenizationConfig(**_BASE, output_name="pan_cancer_ctrcd_v3_cardiac_labs", cardiac_labs_only=True),
]
```

### CLI

```bash
python tokenization_src/tokenize_cli.py \
    --data-dir <DATA_DIR> \
    --cohort pan_cancer_ctrcd_v3 \
    --name pan_cancer_ctrcd_v3_all_labs
```

Add `--summarize` to print cohort statistics and save figures in the same command.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | yes | — | Path to the MIMIC-IV raw data directory |
| `--cohort` | no | `cycle_modeling_ver2` | Source cohort directory under `cohort_outputs/` |
| `--name` | no | `ver1` | Output subdirectory under `tokenization_outputs/` |
| `--max-seq-len` | no | `600` | Maximum token sequence length (truncates oldest events) |
| `--summarize` | no | off | Print summary statistics and save figures after tokenizing |
| `--insert-att` | no | off | Insert CEHR-BERT Artificial Time Tokens (`W0`–`W3`, `M1`–`M11`, `LT`) between consecutive visits |
| `--insert-visit-delimiters` | no | off | Wrap each visit's events with `[V_START]`/`[V_END]` tokens |
| `--bucket-labs` | no | off | Append per-itemid quantile bucket (`_Q1`–`_Q4`) to lab tokens — changes vocab |
| `--bucket-medications` | no | off | Append per-drug dose-tier bucket (`_Q1`–`_Q4`) to medication tokens — changes vocab |
| `--only-abnormal-labs` | no | on (default) | Include only flagged-abnormal lab results |
| `--include-all-labs` | no | off | Include all lab results regardless of the MIMIC abnormality flag; mutually exclusive with `--only-abnormal-labs` |

**Lab filtering note:** By default only labs with `flag IS NOT NULL` in MIMIC `labevents` are tokenized. Use `--include-all-labs` to include every inpatient result regardless of flag. The resolved flag is written to `metadata.json` for provenance.

**Outputs** (`tokenization_outputs/<name>/`):

| File | Description |
|---|---|
| `concept_ids.pt` | Long tensor `(N, max_seq_len)` |
| `type_ids.pt` | Long tensor `(N, max_seq_len)` |
| `visit_ids.pt` | Long tensor `(N, max_seq_len)` |
| `position_ids.pt` | Long tensor `(N, max_seq_len)` |
| `dates.pt` | Long tensor `(N, max_seq_len)` — days since 2000-01-01 per token |
| `age_ids.pt` | Long tensor `(N,)` — decade bucket (0–9) |
| `age_years.pt` | Float tensor `(N,)` — continuous age in years |
| `labels.pt` | Long tensor `(N,)` — binary labels |
| `attention_mask.pt` | Bool tensor `(N, max_seq_len)` |
| `samples.csv/.parquet` | Per-sample metadata (subject_id, cycle_number, prediction_time, binary_label, seq_len) |
| `vocab.json` | Concept and type vocabulary mappings |
| `metadata.json` | `max_seq_len`, `positive_rate`, vocab size, tokenisation flags |

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

ATT token thresholds:

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
python tokenization_src/summarize_tokenization.py tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs
```

**Figures saved** (`tokenization_outputs/<name>/summarization_figures/`):

| File | Description |
|---|---|
| `label_distribution.png` | Positive / negative sample counts (sample- and patient-level) |
| `sequence_length_histogram.png` | Distribution of sequence lengths with truncation line |
| `vocabulary_breakdown.png` | Token counts by event type |
| `age_distribution.png` | Distribution of patient age decade buckets |
| `drug_cycles_distribution.png` | Cycle number histogram (sample-level) and max cycles per patient (patient-level) |

---

## Step 4 — Train

Two model architectures are available:

| `model_type` | Class | Complexity | File |
|---|---|---|---|
| `"transformer"` (default) | `EHR_Encoder` | O(L²) attention | `model_src/ehr_encoder.py` |
| `"mamba"` | `EHR_Mamba` | O(L) SSM recurrence | `model_src/ehr_mamba.py` |

Both share the same `EHR_Event_Embedding` layer and the same ablation flags (`fusion`, `use_time`, `use_age`). `model_type="mamba"` requires CUDA and `mamba-ssm` installed.

```bash
python model_src/train.py \
    --data-dir tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs \
    --output-dir experiment_outputs/run1
```

Quick debug run:

```bash
python model_src/train.py \
    --data-dir tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs \
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
| `--label-smoothing` | `0.0` | Label smoothing for CrossEntropyLoss |
| `--d-model` | `128` | Embedding / hidden dimension |
| `--num-heads` | `8` | Attention heads (must divide `d-model`) |
| `--num-layers` | `2` | Number of TransformerEncoder layers |
| `--ff-dim` | `256` | Feed-forward inner dimension |
| `--dropout` | `0.1` | Dropout probability |
| `--num-workers` | `0` | DataLoader worker processes |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--seed` | `42` | Controls both the patient split and weight initialisation |
| `--use-wandb` | off | Enable Weights & Biases experiment tracking |
| `--fusion` | `add` | `add`: BEHRT-style sum. `concat`: CEHR-BERT concat→Linear→GELU |
| `--use-time` | off | Add sinusoidal time-gap embedding per token |
| `--use-age` | off | Add continuous-age sinusoidal embedding |

**Outputs** (`experiment_outputs/<run>/`):

| File | Description |
|---|---|
| `best_model_auroc.pt` | Checkpoint with best val AUROC |
| `best_model_auprc.pt` | Checkpoint with best val AUPRC |
| `best_model_f1.pt` | Checkpoint with best val F1 |
| `best_model_sensitivity.pt` | Checkpoint with best val sensitivity |
| `best_model_specificity.pt` | Checkpoint with best val specificity |
| `config.json` | All hyperparameters, derived vocab/seq sizes, hardware info, and run date |
| `history.json` | Per-epoch train loss, val loss, val AUROC, AUPRC, F1, sensitivity, specificity, and elapsed time |
| `test_metrics.json` | Test results for the AUROC checkpoint (backward-compatible) |
| `test_metrics_{metric}.json` | Test results for each per-metric checkpoint |

Each epoch, the training loop saves a checkpoint whenever a metric improves. After training, all 5 checkpoints are evaluated on the held-out test set and results are printed as a comparison table.

---

## Step 5 — Interpret (xAI)

Post-training interpretability using three complementary techniques:

| Technique | What it shows | Strength |
|---|---|---|
| Raw attention | Per-head `(seq × seq)` weight matrix | Fast; exploratory — attention ≠ attribution |
| Attention rollout | Cross-layer aggregated CLS relevance (Abnar & Zuidema 2020) | More principled than single-layer attention |
| Integrated Gradients | Token attribution satisfying the completeness axiom (Captum) | Gold standard; requires `pip install captum` |

**IG baseline:** Always a zero-embedding sequence — not the `[PAD]` token embedding, but a literal zero tensor in post-LayerNorm embedding space. Represents a neutral "no information" input.

### Single sample via CLI

```bash
# Explain by subject_id and cycle_number (recommended — named output folder)
python interpretation/interpret.py \
    --model-dir experiment_outputs/run1 \
    --data-dir  tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs \
    --subject-id 12345 --cycle-number 2

# Explain by dataset row index
python interpretation/interpret.py \
    --model-dir experiment_outputs/run1 \
    --data-dir  tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs \
    --sample-idx 42

# More accurate IG (default 100 steps; convergence delta < 0.01 is good)
python interpretation/interpret.py ... --ig-steps 200

# Skip IG if captum is not installed
python interpretation/interpret.py ... --skip-ig

# Use the sensitivity-optimised checkpoint instead of the default AUROC one
python interpretation/interpret.py ... --checkpoint-metric sensitivity
```

**Outputs** (`interpretation/outputs/<subject>_cycle<n>/`):

| File | Description |
|---|---|
| `attention_L{i}_H{j}.png` | Attention heatmap for layer `i`, head `j` (non-padding tokens only; CLS row highlighted) |
| `attention_rollout.png` | Top-K tokens by rollout relevance, coloured by event type |
| `integrated_gradients.png` | Top-K tokens by IG attribution (L2 norm over `d_model`), with convergence delta |
| `attributions.csv` | Per-token rollout + IG scores for all non-padding positions, sorted by IG |

### Visualize attributions separately

```bash
python interpretation/visualize_attributions.py \
    --input-dir interpretation/outputs/12345_cycle2
```

| File | Description |
|---|---|
| `comparison_top{K}.png` | IG (top) and rollout (bottom) for the same top-K tokens on aligned panels |
| `event_type_breakdown.png` | Total attribution summed by event type |
| `rollout_vs_ig_scatter.png` | Per-token scatter: rollout vs IG score; top-5 IG tokens annotated |

---

## Model architecture

### Shared embedding layer (`model_src/embedding_layers.py`)

`EHR_Event_Embedding` is controlled by three orthogonal flags:

| Flag | Values | Effect |
|---|---|---|
| `fusion` | `"add"` (default) | BEHRT-style element-wise sum of all embedding tables |
| | `"concat"` | CEHR-BERT: `cat([concept, time*, age*, position]) → Linear(4d→d) → GELU`, then type/visit/segment as additive residuals |
| `use_time` | `False` / `True` | Learned sinusoidal time embedding. Requires `dates.pt`. |
| `use_age` | `False` / `True` | Continuous-age sinusoidal embedding. Requires `age_years.pt`. |

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

### Transformer encoder (`model_type="transformer"`, default)

```
EHR_Event_Embedding → N × TransformerEncoderLayer (pre-norm, GELU FFN) → CLS pooling → Linear(d → 2)
```

- Optimizer: AdamW
- Scheduler: CosineAnnealingLR (`T_max=epochs`, `eta_min=lr/10`)
- Loss: CrossEntropyLoss with inverse-frequency class weights + optional label smoothing
- Mixed precision: `torch.amp.autocast` + `GradScaler` (CUDA only)
- Five checkpoints saved per run: one per metric (`best_model_{auroc/auprc/f1/sensitivity/specificity}.pt`)

**`return_attention` hook:** `EHR_Encoder.forward()` accepts `return_attention=True`, which returns `(logits, all_attn)` where `all_attn` is a list of `(B, num_heads, seq, seq)` tensors — one per layer.

### Mamba encoder (`model_type="mamba"`)

```
EHR_Event_Embedding → N × BiMambaBlock (fwd SSM + bwd SSM + merge) → CLS pooling → Linear(d → 2)
```

`BiMambaBlock` runs two `mamba_ssm.Mamba` instances in opposite directions. Requires CUDA: `pip install causal-conv1d mamba-ssm`.

### Regularization (`run_train.py` defaults for Jul24 data)

The dataset is small (~1,800 training samples). Jul24 defaults are tuned for higher positive rate (35.6%) and label noise from the death endpoint.

| Setting | Default | Sweep range | Notes |
|---|---|---|---|
| `num_layers` | 1 (S) | 1–4 (arch sweep) | Reduced from CEHR-BERT's 5 for dataset size |
| `ff_dim` | 128 (S) | 128–512 (arch sweep) | 2× `d_model` |
| `dropout` | 0.4 | 0.2–0.6 (dropout sweep) | Bumped from 0.3 due to noisier Jul24 labels |
| `weight_decay` | 5e-2 | 1e-2–4e-1 (wd sweep) | Range shifted up vs Jul17; overfitting is primary concern |
| `label_smoothing` | 0.1 | 0.05–0.25 (ls sweep) | Penalises overconfident predictions on noisy labels |
| `lr` | 1e-4 | 5e-5–1e-3 (lr sweep) | AdamW learning rate |

---

## Embedding ablations

| ID | `fusion` | `use_time` | `use_age` | Purpose |
|---|---|---|---|---|
| A0 | `"add"` | False | False | Baseline — no temporal info |
| A1 | `"add"` | True | False | Does time-gap signal help? |
| A2 | `"add"` | False | True | Does patient age help? |
| A3 | `"add"` | True | True | Best additive temporal version |
| B0 | `"concat"` | False | False | Tests concat fusion alone |
| B1 | `"concat"` | True | False | Concat + time |
| B2 | `"concat"` | True | True | CEHR-BERT style |
| C1 | best of A/B | — | — | Same flags + `insert_att=True` tokenization |
| C2 | `"concat"` | True | True | Full CEHR-BERT (concat + time + age + ATT) |

`dates.pt` and `age_years.pt` are **always written** by the tokenizer — no special flag needed for A0–B2. C1/C2 require a separate tokenization built with `insert_att=True`.

---

## Runner scripts

Config values live directly in each runner script — edit the variables at the top, then run.

### `run_cohort.py`

Set `PIPELINE` and `OUTPUT_NAME` at the top:

```python
PIPELINE    = "pan_cancer_ctrcd"
OUTPUT_NAME = "pan_cancer_ctrcd_v3"
```

Override `CYCLE_SQL_DIR` / `PRESCRIPTIONS_SQL_DIR` only when pointing at a non-default SQL subdirectory.

```bash
python run_cohort.py
```

### `run_tokenization.py`

Set `cohort_name` in `_BASE` and add `TokenizationConfig` entries to `RUNS`:

```python
_BASE = dict(cohort_name="pan_cancer_ctrcd_v3", ...)
RUNS  = [TokenizationConfig(**_BASE, output_name="pan_cancer_ctrcd_v3_all_labs")]
```

```bash
python run_tokenization.py
```

### `run_pipeline.py`

Runs cohort + tokenization in sequence. Toggle `RUN_COHORT` / `RUN_TOKENIZE` flags to skip stages.

```bash
python run_pipeline.py
```

### `run_train.py`

Dataset sweep across all four `pan_cancer_ctrcd_v3` tokenization variants using architecture S (small model). Hyperparameters are fixed from the Jul24 sweep findings.

Each `seed` independently controls:
- **Patient split** — which patients go to train / val / test (70 / 15 / 15)
- **Model initialisation** — weight init and dropout randomness

**Sweep structure** (3 seeds each):

| Dataset variant | Tokenization dir | Runs |
|---|---|---|
| `all_labs` | `Jul30_pan_cancer_ctrcd_v3_all_labs` | 3 |
| `cardiac_labs` | `Jul30_pan_cancer_ctrcd_v3_cardiac_labs` | 3 |
| `bucketed_all_labs` | `Jul30_pan_cancer_ctrcd_v3_bucketed_all_labs` | 3 |
| `bucketed_cardiac_labs` | `Jul30_pan_cancer_ctrcd_v3_bucketed_cardiac_labs` | 3 |
| **Total** | | **12** |

Architecture S: `d_model=64, num_heads=4, num_layers=1, ff_dim=128`. Fixed hyperparameters: `lr=1e-4, weight_decay=5e-2, dropout=0.4, label_smoothing=0.1`. Outputs write to `experiment_outputs/pan_cancer_ctrcd/small_model/`.

```bash
python run_train.py
```

### `run_comparisons.py`

Generates all sweep comparison plots in one command. Variants with missing results are skipped automatically — safe to run incrementally while training is in progress.

```bash
python run_comparisons.py              # all sweeps
python run_comparisons.py --sweep arch lr   # specific sweeps only
python run_comparisons.py --show            # display instead of saving
```

Figures are saved to `experiment_outputs/pan_cancer_ctrcd/comparisons/`.

### `run_mamba.py`

Mirror of `run_train.py` for the Mamba encoder. Requires CUDA + `pip install causal-conv1d mamba-ssm`.

```bash
python run_mamba.py
```

### `run_xgboost.py`

XGBoost baseline using hand-crafted features derived from the binary modeling table. Does not require a tokenization step — reads directly from `cohort_outputs/<name>/final_cycle_binary_modeling_table.csv`.

```bash
python run_xgboost.py
```

### `run_interpretation.py`

Batch interpretability runner. Define samples in `RUNS`, then:

```bash
python run_interpretation.py
```

```python
_BASE = dict(
    model_dir     = REPO_ROOT / "experiment_outputs" / "run1",
    data_dir      = REPO_ROOT / "tokenization_outputs" / "Jul30_pan_cancer_ctrcd_v3_all_labs",
    ig_steps      = 100,
    top_k         = 30,
    skip_ig       = False,
    run_visualize = True,
    device        = "auto",
)

RUNS = [
    InterpretationConfig(**_BASE, subject_id=10006008, cycle_number=1),
    InterpretationConfig(**_BASE, subject_id=10006008, cycle_number=2),
]
```

`InterpretationConfig` fields:

| Field | Default | Description |
|---|---|---|
| `model_dir` | required | Experiment directory with `config.json` + `best_model_*.pt` checkpoints |
| `sample_idx` | `None` | Row index in the tokenized dataset |
| `subject_id` | `None` | MIMIC subject_id (use with `cycle_number`) |
| `cycle_number` | `None` | Chemotherapy cycle number (use with `subject_id`) |
| `data_dir` | `None` | Tokenization directory (falls back to value in `config.json`) |
| `output_dir` | auto | Defaults to `interpretation/outputs/<subject>_cycle<n>/` |
| `ig_steps` | `100` | Integration steps for IG (more = more accurate, slower) |
| `skip_ig` | `False` | Skip IG if captum is not installed |
| `top_k` | `30` | Top-K tokens in bar charts |
| `run_visualize` | `True` | Run `visualize_attributions` after `interpret` |
| `checkpoint_metric` | `"auroc"` | Which checkpoint to load (`best_model_{metric}.pt`) |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, or `mps` |

---

## Data exploration

Scripts and notebooks in `data_exploration/`.

### `data_exploration/inspect_patient.py`

Visualize a single patient's tokenized EHR sequence in the terminal, and optionally run a model prediction.

```bash
python data_exploration/inspect_patient.py --patient-idx 5
python data_exploration/inspect_patient.py --subject-id 13595646
python data_exploration/inspect_patient.py --patient-idx 5 --model-dir experiment_outputs/run1
```

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `tokenization_outputs/ver1` | Tokenization directory |
| `--split` | `test` | Which split to sample from |
| `--patient-idx` | random | 0-based index within the split |
| `--subject-id` | — | MIMIC subject_id (searches across all splits) |
| `--cycle-idx` | `0` | Which cycle to show for multi-cycle patients |
| `--model-dir` | — | Experiment dir for an attached model prediction |
| `--max-per-visit` | `20` | Max events shown per visit (`0` = all) |

Requires `rich`: `pip install rich`

---

## Evaluation

Post-training evaluation scripts in `evaluation/`.

### `evaluation/evaluate_model.py`

Runs a trained checkpoint on a full data split and reports aggregate metrics and a per-sample table.

```bash
python evaluation/evaluate_model.py --model-dir experiment_outputs/run1
python evaluation/evaluate_model.py --model-dir experiment_outputs/run1 --split val
```

Reports: AUROC, AUPRC, F1, sensitivity, specificity, accuracy, precision, confusion matrix.

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | required | Experiment dir with `config.json` + `best_model_*.pt` checkpoints |
| `--data-dir` | from config | Tokenization directory |
| `--split` | `test` | `train`, `val`, or `test` |
| `--batch-size` | `32` | Inference batch size |
| `--threshold` | `0.5` | Decision threshold for F1 / sensitivity / specificity |
| `--checkpoint-metric` | `auroc` | Which checkpoint to load (`best_model_{metric}.pt`) |
| `--output-csv` | — | Optional path to save per-sample results |

### `evaluation/compare_ablations.py`

Scans an experiment directory for test metrics files, groups by ablation ID, and reports mean ± std across seeds for all metrics with a bar chart.

```bash
python evaluation/compare_ablations.py experiment_outputs/pan_cancer_ctrcd/small_model/
python evaluation/compare_ablations.py experiment_outputs/pan_cancer_ctrcd/small_model/ --sort auprc --metric auprc
```

Expected layout:
```
experiment_outputs/pan_cancer_ctrcd/small_model/
  all_labs/seed42/test_metrics.json
  all_labs/seed52/test_metrics.json
  ...
```

### `evaluation/plot_history.py`

Plots training loss and per-metric validation curves from `history.json`. Pass a variant directory to aggregate across seeds (mean ± std band), or a single seed directory for no aggregation.

```bash
python evaluation/plot_history.py \
    --model-dir experiment_outputs/pan_cancer_ctrcd/small_model/all_labs

python evaluation/plot_history.py \
    --model-dir experiment_outputs/pan_cancer_ctrcd/small_model/all_labs \
                experiment_outputs/pan_cancer_ctrcd/small_model/cardiac_labs \
    --save dataset_sweep.png
```

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | required | Variant dir (aggregates `seed*/history.json`) or single seed dir |
| `--metrics` | `auroc auprc f1` | Which validation metrics to plot |
| `--save` | — | Save figure to this path (PNG/PDF/SVG) |
| `--dpi` | `150` | Output DPI when saving |

---

## Smoke tests

```bash
python model_src/embedding_layers.py        # embedding layer
python model_src/ehr_encoder.py             # transformer architecture + return_attention
python model_src/ehr_mamba.py               # Mamba architecture (requires CUDA + mamba-ssm)
python model_src/dataset.py tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs   # DataLoader
```

---

## Repository structure

```
MIMIC-Cardio-Oncology/
├── cohort_src/
│   ├── cohort_cli.py                      CLI entry point for cohort generation
│   └── generate_cycle_modeling_table.py   DuckDB SQL chain → binary modeling table
├── configs/
│   ├── cohort_config.py                   CohortConfig dataclass
│   ├── tokenization_config.py             TokenizationConfig dataclass
│   ├── train_config.py                    TrainConfig dataclass
│   └── interpretation_config.py          InterpretationConfig dataclass
├── data_exploration/
│   ├── inspect_patient.py                 Terminal patient sequence viewer
│   └── *.ipynb                            Exploratory notebooks
├── evaluation/
│   ├── evaluate_model.py                  Per-split metrics + per-sample table
│   ├── compare_ablations.py               Mean ± std across seeds, all metrics
│   └── plot_history.py                    Training curves (mean ± std across seeds)
├── interpretation/
│   ├── interpret.py                       Attention heatmaps + rollout + Integrated Gradients
│   └── visualize_attributions.py          Summary plots from attributions.csv
├── model_src/
│   ├── embedding_layers.py                EHR_Event_Embedding + TimeEmbeddingLayer
│   ├── ehr_encoder.py                     EHR_Encoder (Transformer, return_attention hook)
│   ├── ehr_mamba.py                       EHR_Mamba (bidirectional Mamba)
│   ├── mamba_embedding.py                 Mamba-specific embedding
│   ├── mamba_train.py                     Mamba training loop
│   ├── train.py                           Transformer training loop
│   └── dataset.py                         EHRDataset + DataLoader helpers
├── sql_files/
│   ├── drug_cycles_sql/
│   │   ├── jul17/                         Jun 23 SQL chain (00→06) — Jul17 training data
│   │   └── jul24/                         Jul 24 SQL chain (00→06) — revised definitions
│   ├── prescriptions_sql/
│   │   ├── jul17/                         Jun 23 prescriptions SQL
│   │   └── jul24/                         Jul 24 prescriptions SQL (expanded drug list)
│   ├── diagnoses_sql/                     Shared across versions
│   └── ...                                Other SQL directories (echo, ecg, procedures)
├── tokenization_src/
│   ├── tokenize_cycle_sequences.py        Core tokenizer
│   ├── tokenize_cli.py                    CLI entry point for tokenization
│   ├── split_dataset.py                   Patient-level stratified split
│   └── summarize_tokenization.py          Summary statistics + figures
├── run_cohort.py                          Runner: cohort generation (set SQL version at top)
├── run_tokenization.py                    Runner: tokenization variants
├── run_pipeline.py                        Runner: cohort + tokenization in one go
├── run_train.py                           Runner: pan_cancer_ctrcd dataset sweep, small model (12 runs)
├── run_train_small.py                     Runner: earlier small-model sweep (v1 datasets, reference)
├── run_comparisons.py                     Runner: generate all sweep comparison plots
├── run_mamba.py                           Runner: Mamba training
├── run_xgboost.py                         Runner: XGBoost baseline
└── run_interpretation.py                  Runner: batch xAI interpretation
```

---

## Typical full run

**Via runner scripts** (edit variables at the top of each file first):

```bash
python run_cohort.py        # builds cohort_outputs/pan_cancer_ctrcd_v3/
python run_tokenization.py  # builds tokenization_outputs/pan_cancer_ctrcd_v3_*/
python run_train.py         # runs 12-run dataset sweep → experiment_outputs/pan_cancer_ctrcd/
python run_interpretation.py
```

Or all data pipeline stages at once:

```bash
python run_pipeline.py
```

**Via CLI:**

```bash
# 1. Build cohort
python cohort_src/cohort_cli.py \
    --data-dir <DATA_DIR> \
    --pipeline pan_cancer_ctrcd \
    --name pan_cancer_ctrcd_v3

# 2. Tokenize
python tokenization_src/tokenize_cli.py \
    --data-dir <DATA_DIR> \
    --cohort pan_cancer_ctrcd_v3 \
    --name pan_cancer_ctrcd_v3_all_labs \
    --max-seq-len 512 \
    --insert-visit-delimiters \
    --include-all-labs \
    --summarize

# 3. Train
python model_src/train.py \
    --data-dir tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs \
    --output-dir experiment_outputs/run1 \
    --seed 42

# 4. Evaluate
python evaluation/evaluate_model.py --model-dir experiment_outputs/run1

# 5. Interpret
python interpretation/interpret.py \
    --model-dir experiment_outputs/run1 \
    --data-dir tokenization_outputs/Jul30_pan_cancer_ctrcd_v3_all_labs \
    --subject-id 10006008 --cycle-number 1
```

### Dataset sweep (12 runs)

```bash
# Run all 4 dataset variants × 3 seeds:
python run_train.py

# Compare test metrics across variants:
python evaluation/compare_ablations.py experiment_outputs/pan_cancer_ctrcd/small_model/ --sort auroc --metric auroc
```

### Adding a new pipeline

1. Define a `PipelineConfig` in `cohort_src/cohort_pipeline.py` and register it in `PIPELINE_REGISTRY`.
2. Create `sql_files/drug_cycles_sql/<pipeline>/` and `sql_files/prescriptions_sql/<pipeline>/` with the new SQL files.
3. Set `PIPELINE` and `OUTPUT_NAME` in `run_cohort.py`, then run `python run_cohort.py` and `python run_tokenization.py`.
