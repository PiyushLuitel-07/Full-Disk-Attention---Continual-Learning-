# Full-Disk Attention: Small Two-Stage EWC Project

This directory is a focused, runnable extension of `../fulldiskattention`. It
keeps the original binary task and attention architecture, but exposes the data
chronologically and adds **Elastic Weight Consolidation (EWC)**.

The implemented experiment is intentionally limited to:

| Stage | Years | What happens |
|---|---|---|
| 1 | 2010–2012 | Train a newly initialized attention model with cross-entropy |
| 2 | 2013–2014 | Load Stage 1 and train with cross-entropy + EWC |

The input is one 256×256 grayscale full-disk HMI magnetogram. The output is
binary: `0 = NF`, `1 = at least one M1.0-or-stronger flare in the next 24 hours`.

Stage 1 is never initialized from the supplied all-year checkpoints. Doing so
would leak 2013–2018 information into the past.

## What was kept and what was changed

Kept from the original repository:

- the six-block convolutional model and three attention estimators;
- grayscale resize-to-256 preprocessing with tensor values in `[0, 1]`;
- FL-only rotations/flips;
- class-balanced training;
- SGD, cross-entropy, TSS, HSS, and threshold 0.5;
- the original seasonal fold as an outer train/evaluation partition.

Added for continual learning:

- year-based Stage 1 and Stage 2 manifests;
- exact per-example empirical diagonal Fisher estimation;
- classic multi-anchor EWC;
- stage-by-stage checkpoints and all-seen-stage evaluation;
- safe one-class metrics, JSON/CSV logs, provenance, and bounded downloads;
- CPU, one-GPU, and normal project-environment support.
- complete Stage 1/2 experiment tracking in Weights & Biases.
- a matched Stage-2-only reference and standard forward-transfer measurement.

Fold 3 is the pilot default because it remains useful in later low-activity
years. A publication experiment should eventually repeat the locked setup over
all four outer folds and multiple seeds.

## Project map

```text
fulldiskattention_continual/
├── configs/                 pilot and real-server JSON settings
├── data_labeling/           chronological manifest generator
├── download_mag/            bounded Helioviewer downloader
├── modeling/                model, loader, metrics, EWC, trainer
├── tools/                   synthetic demo generator and run summary
├── tests/                   small unit/model tests
├── server/                  scheduler examples (not NOVA assumptions)
├── data/                    generated manifests/images
├── runs/                    generated checkpoints/results
├── CODE_WALKTHROUGH.md      detailed explanation of the code
└── requirements.txt
```

## Run the complete tiny demo today

The demo uses real 2010–2014 label rows but creates synthetic images. It tests
the software and gives intuition; it is **not a solar-flare result**.

From this directory:

```bash
source ../fulldiskattention/.venv/bin/activate

python data_labeling/make_stage_manifests.py \
  --fold 3 \
  --output-dir data/manifests/fold3_pilot \
  --max-train-per-class 4 \
  --max-eval-per-class 4 \
  --fisher-per-class 2

python tools/make_demo_images.py \
  --manifest-dir data/manifests/fold3_pilot \
  --image-root data/demo_images

python -m unittest discover -s tests -v

python -m modeling.train_continual \
  --config configs/pilot.json \
  --stage 1

python -m modeling.train_continual \
  --config configs/pilot.json \
  --stage 2

python tools/summarize_run.py --run-dir runs/pilot_ewc
```

Run the matched no-EWC pilot with the same data, seed, model, and training
budget:

```bash
python -m modeling.train_continual \
  --config configs/pilot_finetune.json \
  --stage 1

python -m modeling.train_continual \
  --config configs/pilot_finetune.json \
  --stage 2

python tools/compare_methods.py \
  --ewc-run-dir runs/pilot_ewc \
  --finetune-run-dir runs/pilot_finetune
```

The fine-tuning baseline loads its Stage 1 checkpoint and learns Stage 2 in the
same order, but applies no EWC penalty and performs no Fisher estimation. The
comparison command refuses to compare runs if their stage definitions, model,
training settings, manifests, image root, fold, or seed differ.

After the EWC Stage 1 run exists, run the Stage-2-only pilot once:

```bash
python -m modeling.train_continual \
  --config configs/pilot_stage2_reference.json \
  --stage 2

cat runs/pilot_stage2_only/metrics/forward_transfer_summary.json
```

This command never loads Stage 1 model weights. It first evaluates the random
model on Stage 2 to obtain `b_2`, trains only with 2013–2014 rows, and evaluates
the trained reference. It reads the EWC Stage 1 summary only to obtain `R_1,2`
and verifies that the seed, data, model, and training controls match. Standard
forward transfer is `FWT = R_1,2 - b_2`; the trained Stage-2-only score is
reported separately and is not substituted for `b_2`.

What to notice:

1. Stage 1 prints `EWC active=False` because no older task exists.
2. Fisher estimation runs after Stage 1 and saves the parameter importance.
3. Stage 2 prints `EWC active=True` and logs a separate weighted EWC term.
4. Both checkpoints are evaluated on both chronological evaluation sets.
5. The Stage 1 checkpoint's Stage 2 result is a zero-shot measurement only;
   Stage 2 images do not update the Stage 1 model.
6. `summarize_run.py` prints the complete two-stage TSS/HSS matrix and
   Stage 1 retention after Stage 2.

## Weights & Biases experiment tracking

All supplied experiment configurations enable W&B. Stage 1 and Stage 2 are
logged as separate runs in one group, so the two commands remain independent
while the continual sequence stays together in the W&B project. Each stage
logs:

- learning rate and training CE, EWC, total-loss, TSS/HSS and confusion metrics
  for every epoch;
- held-out evaluation metrics and a confusion-matrix chart;
- Fisher coverage, importance statistics and a per-parameter Fisher table;
- the complete experiment configuration, runtime metadata and elapsed time;
- one versioned artifact containing checkpoints, EWC state, histories,
  predictions and provenance through the completed stage.

Authenticate once on NOVA inside your account:

```bash
wandb login
```

Do not put the API key in a JSON file, shell script, Git commit, or README. W&B
uses the credentials already stored by `wandb login`. The configured entity and
project are:

```text
piyush-luitel-texas-christian-university
Full disk attention solar flare prediction with continual learning
```

The local `wandb/` cache is ignored by Git. Set `tracking.mode` to `offline`
only when NOVA cannot reach W&B; later upload such a run with `wandb sync`.

The pilot deliberately processes only two training batches and four Fisher
examples per stage. Its scores are unstable and scientifically meaningless.

To repeat it, use a new `paths.run_dir` in a copied configuration. The trainer
refuses to overwrite existing stage artifacts.

## Prepare the full real-data manifests

This is fast and does not need the images:

```bash
python data_labeling/make_stage_manifests.py \
  --fold 3 \
  --output-dir data/manifests/fold3 \
  --fisher-per-class 64
```

The source is the original repository's committed
`Fold3_train.csv`/`Fold3_val.csv`. The generated `summary.json` records source
and output hashes, class counts, and year ranges. A Fisher set of 64 examples
per class is a cautious first server run; increase toward 1,024 per class only
after timing the exact per-example gradient calculation.

## Where to get the HMI images

Best option: obtain the original 512×512 JPEG corpus from the professor or
original project storage. Put it beneath one root with paths like:

```text
hmi_jpgs_512/2011/02/15/HMI.m2011.02.15_01.00.00.jpg
```

This is best for comparison because it preserves the original bytes and
JP2-to-JPEG conversion.

If that corpus is unavailable, download only manifest rows from the official
[Helioviewer API v2](https://api.helioviewer.org/docs/v2/) service. The official
data-source table identifies source ID 19 as SDO/HMI magnetograms. First test
the pilot set:

```bash
python download_mag/download_from_manifests.py \
  --manifest data/manifests/fold3_pilot/stage1_train.csv \
             data/manifests/fold3_pilot/stage1_eval.csv \
             data/manifests/fold3_pilot/stage2_train.csv \
             data/manifests/fold3_pilot/stage2_eval.csv \
  --image-root data/hmi_jpgs_512 \
  --jp2-root data/jp2 \
  --provenance data/download_provenance.csv
```

Then validate that Pillow can decode JPEG2000:

```bash
python -c "from PIL import features; print(features.check('jpg_2000'))"
```

Only after the small download succeeds should you point the command at the full
manifests. Expect tens of thousands of network requests for a full two-stage
Fold 3 run. Keep the provenance CSV. The new converter is consistent within
this project but may not be pixel-identical to the original OpenCV conversion,
so state which image source was used in every result.

Do not run `../fulldiskattention/download_mag/download_jp2.py` unchanged: it
attempts an open-ended multi-year 12-minute-cadence download.

## First real Stage 1 and Stage 2 run

Copy both server templates to untracked working configurations:

```bash
cp configs/server_template.json configs/server_ewc.json
cp configs/server_finetune_template.json configs/server_finetune.json
cp configs/server_stage2_reference_template.json configs/server_stage2_reference.json
```

Make the same image-path and training-budget edits in all three files. Keep
separate run directories. In the Stage-2 reference config,
`source_continual_run_dir` must point to the EWC run. The code verifies the
controlled settings before reporting EWC or forward-transfer results.

Recommended first intuition run:

- `epochs_per_stage`: 1;
- `max_batches_per_epoch`: 20;
- `batch_size`: as large as one GPU safely permits;
- `maximum_total_fisher_examples`: 16.

After it works, create a new run directory and increase the budget gradually.
Do not interpret a partial run as the final experiment.

Train Stage 1 first:

```bash
python -m modeling.train_continual \
  --config configs/server_ewc.json \
  --stage 1 \
  --device cuda
```

Inspect `runs/<run>/metrics/stage1_summary.json` and the Fisher summary. Then:

```bash
python -m modeling.train_continual \
  --config configs/server_ewc.json \
  --stage 2 \
  --device cuda
```

Stage 2 loads only the previous-stage artifacts in the same run directory.
Its training rows are 2013–2014 only. Old images are not replayed.

Then run the same two commands with `configs/server_finetune.json`. After both
sequences finish:

```bash
python tools/compare_methods.py \
  --ewc-run-dir runs/fold3_seed4_ewc_lambda10 \
  --finetune-run-dir runs/fold3_seed4_finetune
```

`forgetting_reduction = finetune_forgetting - EWC_forgetting`; a positive value
favors EWC. Always inspect `stage2_performance_difference` alongside it because
retention is not useful if EWC prevents learning the new stage.

Finally run the reference directly at Stage 2; there is no Stage 1 command for
this method:

```bash
python -m modeling.train_continual \
  --config configs/server_stage2_reference.json \
  --stage 2 \
  --device cuda

cat runs/fold3_seed4_stage2_only/metrics/forward_transfer_summary.json
```

This is a separate training run on NOVA. It uses only Stage 2 training images,
creates no Fisher state, and logs the random baseline, training curves, final
Stage 2 evaluation, and TSS/HSS forward transfer to W&B.

## Running on NOVA safely

The two supplied NOVA administrator PDFs are identical. They say to prefer a
user/project environment for individual research. You do not need shared
`/opt/nova` software or application-administrator privileges for this project.
Do not use `sudo pip` or modify the NVIDIA driver/CUDA installation.

After login, inspect the machine before deciding how to run:

```bash
hostname
whoami
pwd
nvidia-smi
python3 --version
df -h .
command -v module
command -v sbatch
command -v srun
command -v tmux
```

Ask the local administrator/professor for the hostname, GPU allocation,
scheduler policy, storage location, and whether training is prohibited on the
login node. The PDF does not specify any of those items.

Create an environment inside the copied project:

```bash
cd /path/to/fulldiskattention_continual
python3 --version  # use Python 3.10 or newer for a fresh current PyTorch setup
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use the official PyTorch selector to install a wheel compatible with the
server's driver, then:

```bash
python -m pip install numpy pillow
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("compiled CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
PY
```

If `sbatch` exists, adapt the files in `server/` to the local policy. They are
examples, not proof that NOVA uses Slurm. Create `logs/` before submission and
make Stage 2 depend on successful Stage 1 completion, or submit it afterward.
If no scheduler exists, use the administrator-approved interactive process
(often `tmux`) rather than assuming a login-node job is acceptable.

## Outputs and how to read them

```text
runs/<run>/
├── checkpoints/stage1.pt, stage2.pt
├── ewc/through_stage1.pt, through_stage2.pt
├── ewc/stage*_fisher_summary.json
├── metrics/history.csv
├── metrics/stage*_summary.json
├── metrics/continual_summary.json
├── metrics/ewc_vs_finetune.json    # created by the comparison command
├── metrics/forward_transfer_summary.json  # Stage-2-only reference run
├── predictions/after_stage*_eval_stage*.csv
├── predictions/random_init_eval_stage2.csv
└── provenance_stage*.json
```

- `history.csv` separates CE, raw EWC, weighted EWC, and total loss.
- `through_stage1.pt` contains Stage 1 anchors and Fisher diagonals.
- The four checkpoint/evaluation prediction files let you recompute every
  entry in the two-stage evaluation matrix.
- forgetting after Stage 2 is `Stage1_TSS_after_stage1 - Stage1_TSS_after_stage2`.

The complete matrix is:

```text
                              evaluate 2010–2012   evaluate 2013–2014
checkpoint after Stage 1             R_1,1                 R_1,2
checkpoint after Stage 2             R_2,1                 R_2,2
```

`R_1,2` measures zero-shot performance before Stage 2 learning. It is recorded
for forward-transfer analysis but is never used to train Stage 1.

After Stage 2, `continual_summary.json`, W&B, and `summarize_run.py` report the
following separately for TSS and HSS:

- `final_average = (R_2,1 + R_2,2) / 2`, performance over both learned stages;
- `average_incremental_performance = (R_1,1 + final_average) / 2`, which also
  accounts for performance after the first learning step;
- `forgetting = R_1,1 - R_2,1`, where a positive value means old skill fell;
- `backward_transfer = R_2,1 - R_1,1`, the signed inverse of forgetting in this
  two-stage experiment;
- `stage2_gain = R_2,2 - R_1,2`, improvement from zero-shot prediction to the
  trained Stage 2 checkpoint.

The Stage-2-only reference additionally reports:

- `b_2`: random-initialization performance on the Stage 2 evaluation rows;
- `forward_transfer = R_1,2 - b_2`, where positive means Stage 1 learning
  improved Stage 2 performance before Stage 2 training;
- `stage2_only_after_training`, a separate scratch-training reference that is
  not part of the standard FWT equation.

Undefined base scores remain `null`; they are not silently replaced with zero.

Classic EWC stores one anchor and one Fisher value per parameter. For this
roughly 7.5-million-parameter model, Stage 1 EWC state is around 60 MB in
float32. This is expected.

## Scientific limits of this first implementation

This small project is a clear starting point, not the final thesis experiment:

- it implements only Stage 1 and Stage 2;
- it uses fixed epochs and does not tune on outer evaluation data;
- the provided lambda `10` is an example, not a validated optimum;
- Fold 3 is only a debugging fold;
- hourly targets overlap and inherit the paper's boundary-correlation issue;
- EWC protects trainable BatchNorm scale/bias, not its running buffers;
- final claims require the matched naive fine-tuning comparison, all four folds,
  and multiple seeds. Joint/offline training is intentionally outside the
  current project scope.

Keep this code stable until the two-stage real-data run is understood. Then add
the later stages or baselines as separate, deliberate experiments.

## License note

This project derives the model architecture from `../fulldiskattention`, which
is distributed under GNU GPL v3. Preserve that repository's `LICENSE` and its
notices when moving or distributing this continual-learning extension.
