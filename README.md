# Forgetting Ledger

**Who made the model forget, where, and what?** A complete and counterfactually testable
attribution of catastrophic forgetting.

When a network learns task *k*, the loss of every previously learned class rises. The
Forgetting Ledger decomposes that realised change — exactly, up to a controlled quadrature
error — into contributions of

* every **new-task training sample** (who caused it),
* every **replayed memory** and **regulariser term** of the continual learner (what protected against it),
* **normalisation running statistics** (BatchNorm drift, measured exactly),
* every **parameter / unit / layer** (where it happened),

for every **old class or task** separately (what was forgotten). All contributions sum to the
measured loss change, and every claim is checked against ground-truth interventions
(removal-and-retrain, linear datamodeling score, class-targeted surgery, rollback,
freeze-and-retrain).

## How it works (one SGD step)

For an objective written as weighted per-sample terms `Σ_i ℓ_i(θ)` and update `Δ = −η P(∇Σ_i ℓ_i)`
(`P` = identity, or A-GEM's projection), the change of an old-group probe loss `L_g` is

```
L_g(θ+Δ, s') − L_g(θ, s) = [L_g(θ, s') − L_g(θ, s)]         running-stats drift  (exact)
                          + ḡ_g · Δ                          parameter path       (quadrature)
ḡ_g = ∫₀¹ ∇L_g(θ + τΔ) dτ     (adaptive composite Simpson, refined until it matches the known endpoint change)
ḡ_g · Δ = Σ_params ḡ_g ⊙ Δ = −η Σ_i ∇ℓ_i · Pᵀ ḡ_g            exact split over parameters and over loss terms
```

Per-term inner products `∇ℓ_i · v` are obtained for all `i` at once with a double-backward
identity, which remains exact under BatchNorm in training mode.

## Install

```bash
git clone https://github.com/Basil-Mohammad/forgetting-ledger.git
cd forgetting-ledger
pip install -r requirements.txt
pytest -q tests            # completeness, exact source split, resume, removal stability (~3 min on CPU)
```

## Run

```bash
# tracked run (resumable: re-running the same command continues from state/latest.pt)
python scripts/run.py train configs/pm5_mlp.yaml seed=0

# the full protocol for one seed: train -> scores -> removal -> surgery -> lds -> params
python scripts/run.py all configs/pm5_mlp.yaml seed=0

# any config value can be overridden on the command line
python scripts/run.py train configs/sc10_resnet.yaml seed=1 learner.name=er model.norm=bn

# aggregate every run into tables (Markdown / JSON) with 95% CIs and Holm-corrected paired tests
python -m flgr.analysis.aggregate --runs ./runs --out ./results
```

CIFAR experiments are meant for a GPU: open `notebooks/colab_cifar.ipynb` (Google Colab) or
`notebooks/kaggle_cifar.ipynb` (Kaggle). Both keep all checkpoints in persistent storage and
resume automatically after a disconnect.

## Benchmarks and learners

| config | scenario | model |
|---|---|---|
| `pm5_mlp.yaml` | Permuted-MNIST, 5 tasks, domain-IL | MLP 784-256-256 |
| `sf2_mlp.yaml` | Split-FashionMNIST, 2 tasks, domain-IL | MLP |
| `sm5_mlp.yaml` | Split-MNIST, 5 tasks, class-IL / task-IL | MLP |
| `sc10_resnet.yaml` | Seq-CIFAR-10, 5 tasks, class-IL / task-IL | reduced ResNet-18 (GN or BN) |
| `sc100_resnet.yaml` | Seq-CIFAR-100, 10 tasks, class-IL / task-IL | reduced ResNet-18 |

Learners: `finetune`, `er`, `derpp`, `ewc` (online), `agem`. Replay uses a *bottom-k hash
buffer* — distributionally a reservoir, but a sample's membership never depends on other
samples, so counterfactual retraining after removals is clean.

## Checkpoints and outputs

Each run directory contains:

```
config.json env.json git.txt metrics.jsonl
state/latest.pt                resumable state (model, learner, ledger accumulators, RNG, position)
snapshots/task{k}_start.pt     full learner state at the start / end of each task
snapshots/task{k}_end.pt
snapshots/task{k}_cp{j}.pt     intra-task parameter checkpoints (TracIn-CP baselines)
ledger/task{k}.pt              ledger tensors: data [N,G], mem, reg, stats, param [P], unit [U,G], layer [L,G], ...
scores/task{k}.pt              baseline attribution scores, single-point explanations, interference map
interventions/*.json           removal, surgery, LDS, rollback, freeze-and-retrain results
eval/task{k}.json              accuracy-matrix row and per-class test metrics
```

## Reproducibility

Every random decision about a sample (split, permutation, visiting order, augmentation, buffer
admission, replay draws) is a pure function of `(seed, stream, task, epoch, index)`. Removing
samples therefore leaves the treatment of all remaining samples unchanged. The experimental
protocol, hypotheses and statistics are fixed in advance in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Repository layout

```
flgr/data.py          scenarios, removal-stable data pipeline, augmentation
flgr/models.py        MLP, reduced ResNet-18 (BN / GN / none), unit and layer indexing
flgr/learners.py      finetune, ER, DER++, online EWC, A-GEM as weighted per-sample terms
flgr/ledger.py        probe sets, adaptive quadrature, exact per-source / per-parameter split
flgr/trainer.py       training loop, checkpoint / resume, snapshots, evaluation
flgr/scores.py        baseline attributions (TracIn-CP, gradient similarity, loss, features, Fisher)
flgr/experiments.py   commands: train, scores, removal, surgery, lds, params
flgr/analysis/        aggregation, statistics, tables
configs/              benchmark configs (YAML, CLI-overridable)
notebooks/            Colab and Kaggle runners for the CIFAR experiments
tests/                correctness tests
pilot/                the original feasibility study
```

## License

MIT
