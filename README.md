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

What to notice:

1. Stage 1 prints `EWC active=False` because no older task exists.
2. Fisher estimation runs after Stage 1 and saves the parameter importance.
3. Stage 2 prints `EWC active=True` and logs a separate weighted EWC term.
4. Stage 2 evaluates both the old 2010–2012 set and new 2013–2014 set.
5. `summarize_run.py` shows retention: Stage 1 TSS before and after Stage 2.

## Weights & Biases experiment tracking

Both supplied configurations enable W&B. Stage 1 and Stage 2 are logged as
separate runs in one group, so the two commands remain independent while the
continual sequence stays together in the W&B project. Each stage logs:

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

Copy the server template to an untracked working config and edit only the paths
and initial training budget:

```bash
cp configs/server_template.json configs/server.json
```

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
  --config configs/server.json \
  --stage 1 \
  --device cuda
```

Inspect `runs/<run>/metrics/stage1_summary.json` and the Fisher summary. Then:

```bash
python -m modeling.train_continual \
  --config configs/server.json \
  --stage 2 \
  --device cuda
```

Stage 2 loads only the previous-stage artifacts in the same run directory.
Its training rows are 2013–2014 only. Old images are not replayed.

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
├── predictions/after_stage*_eval_stage*.csv
└── provenance_stage*.json
```

- `history.csv` separates CE, raw EWC, weighted EWC, and total loss.
- `through_stage1.pt` contains Stage 1 anchors and Fisher diagonals.
- Stage 2 prediction files let you recompute every confusion matrix.
- forgetting after Stage 2 is `Stage1_TSS_after_stage1 - Stage1_TSS_after_stage2`.

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
- final claims require naive fine-tuning and joint/offline comparisons, all four
  folds, and multiple seeds.

Keep this code stable until the two-stage real-data run is understood. Then add
the later stages or baselines as separate, deliberate experiments.

## License note

This project derives the model architecture from `../fulldiskattention`, which
is distributed under GNU GPL v3. Preserve that repository's `LICENSE` and its
notices when moving or distributing this continual-learning extension.
