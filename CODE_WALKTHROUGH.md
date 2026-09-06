# Code Walkthrough: What Happens During Continual Learning

This document explains the shortest path through the code. Read it alongside
the terminal output from the pilot.

## 1. Data is split by two different ideas

The original Fold 3 CSVs still decide which rows are training rows and which
are untouched outer evaluation rows. Calendar years decide the continual
stage:

```text
original Fold3_train ∩ years 2010–2012 -> stage1_train.csv
original Fold3_val   ∩ years 2010–2012 -> stage1_eval.csv

original Fold3_train ∩ years 2013–2014 -> stage2_train.csv
original Fold3_val   ∩ years 2013–2014 -> stage2_eval.csv
```

`data_labeling/make_stage_manifests.py` performs only this intersection. It
does not recreate or alter the original GOES labels. It asserts that fold
training/evaluation paths are disjoint, stages do not overlap, Fisher rows come
only from that stage's training rows, and every target is binary.

The Fisher CSV is a small, deterministic, class-balanced subset of the current
stage's training CSV. Evaluation images never enter Fisher estimation.

## 2. An image becomes a model input

`modeling/dataset.py` joins `image_root` with the relative `label` value, opens
the JPEG explicitly as grayscale, resizes it to 256×256, and converts it to a
tensor shaped `[1, 256, 256]` with values in `[0, 1]`.

The training sampler gives NF and FL equal sampling probability. Only FL
examples may be unchanged, horizontally flipped, vertically flipped, or
rotated by at most five degrees. Evaluation and Fisher transforms are always
deterministic and never augmented.

## 3. The attention model is the original scientific model

`modeling/attention_model.py` has six convolution blocks. Features from blocks
3, 4, and 5 preserve three spatial resolutions. Each attention estimator learns
one spatial compatibility value per location, normalizes those values with a
softmax, and forms a 512-value weighted feature summary.

The three summaries are concatenated into 1,536 values and the final linear
layer produces two logits:

```text
magnetogram -> convolutional features -> 3 attention summaries
             -> concatenate -> [NF logit, FL logit]
```

Softmax converts the logits into probabilities. Class 1 probability at least
0.5 becomes an FL prediction.

## 4. Stage 1 is ordinary supervised learning

For one Stage 1 batch, `modeling/train_continual.py` calculates:

```text
total loss = cross-entropy
```

There is no EWC term because nothing older exists to protect. After the fixed
training budget, the code evaluates the model on `stage1_eval.csv` and writes
one prediction per image.

## 5. Fisher asks which parameters mattered

After Stage 1, `modeling/ewc.py` puts the model in evaluation mode. For each
Fisher example separately it computes the log probability of the true class
and differentiates it with respect to every trainable parameter.

For parameter scalar `i`:

```text
F_i = average over examples of (gradient_i of log p(true class | image))²
```

Squaring makes every importance non-negative. A high value means that changing
that scalar is more likely to disturb Stage 1 behavior. The calculation is
per-example on purpose; squaring one summed batch gradient would introduce
cross-example terms.

The code then stores:

- `anchor_i`: the learned Stage 1 parameter value;
- `F_i`: the Stage 1 Fisher importance.

Both are detached float32 tensors and cannot receive future gradients.

## 6. Stage 2 adds the EWC protection

Stage 2 constructs the same model and loads Stage 1 weights. It loads the
Stage 1 anchor/Fisher, but its image loader contains only 2013–2014 rows. There
is no replay of old images.

For every Stage 2 batch:

```text
raw EWC = sum_i F_i * (current_i - anchor_i)²

total loss = Stage2 cross-entropy + 0.5 * lambda * raw EWC
```

Immediately at the Stage 1 anchor, raw EWC is zero. As Stage 2 gradients move
important parameters, the penalty becomes positive. `lambda` controls the
stability/plasticity trade-off:

- small lambda: easier Stage 2 adaptation, potentially more Stage 1 forgetting;
- large lambda: stronger retention, potentially weaker Stage 2 learning.

The supplied `lambda=10` is only a smoke-test setting. It must not be presented
as scientifically selected.

### Matched fine-tuning control

When `experiment.method` is `finetune`, Stage 1 and Stage 2 use the identical
model, chronological data, seed, optimizer, augmentation, and training budget.
Stage 2 still loads the Stage 1 checkpoint, but the EWC penalty is disabled.
Because Fisher information cannot influence this baseline, the trainer skips
Fisher estimation and does not save an EWC state. This makes fine-tuning the
direct control needed to determine whether EWC actually reduces forgetting.

## 7. Evaluation measures both learning and forgetting

After Stage 2, the model is evaluated independently on:

- Stage 1 held-out data: did old skill survive?
- Stage 2 held-out data: did the model learn the new period?

`modeling/metrics.py` calculates TP, FP, TN, FN, TSS, HSS, recall, precision,
false-positive rate, and accuracy. TSS is:

```text
TSS = TP / (TP + FN) - FP / (FP + TN)
```

When a class is absent, a mathematically undefined metric becomes `NaN`
internally and `null` in JSON instead of crashing or silently inventing zero.

The simple old-stage forgetting measurement is:

```text
Stage 1 forgetting = TSS after Stage 1 - Stage 1 TSS after Stage 2
```

EWC has helped only if it forgets less than a matched naive-fine-tuning run
without destroying current-stage learning. The synthetic pilot cannot answer
that scientific question; it only proves that the mechanism executes.

## 8. Why each artifact exists

- checkpoint: restores model and optimizer/scheduler state;
- EWC state: contains anchors/Fishers needed by the next stage;
- history: shows whether the EWC term was active and its scale relative to CE;
- prediction CSV: makes all metrics independently reproducible;
- Fisher summary: exposes zero/non-zero counts and layer importance magnitude;
- provenance: records the command and software/device versions.

No output from the original repository is overwritten.
