# Experimental Protocol (pre-registered)

This document fixes the research questions, settings, baselines, metrics and
statistical procedure **before** the main experiments are run. Every table and
figure in `paper/sections/results.tex` maps to exactly one experiment ID below.
Deviations from this protocol must be logged in the "Deviations" section at the
end, with a date and a reason.

---------------------------------------------------------------------------

## 0. Object of study

A continual learner is trained on tasks `T_1, …, T_K`. During the training of
task `T_k` (`k ≥ 2`), the loss of every *old group* `c` (an old class, or an old
task) on a held-out **probe set** rises. The **Forgetting Ledger** decomposes the
realised change of each old-group loss,

    ΔL_c = L_c(θ_end, s_end) − L_c(θ_start, s_start),

into contributions that sum to it (up to integration error):

| source                         | granularity                      | tensor                 |
|--------------------------------|----------------------------------|------------------------|
| new-task training samples      | per sample × per old group       | `C_data [N_k, G]`      |
| replayed memory samples        | per memory slot × per old group  | `C_mem  [M, G]`        |
| regularisation terms (EWC, …)  | per term × per old group         | `C_reg  [R, G]`        |
| normalisation running stats    | per step × per old group         | `C_stats [G]`          |
| parameters                     | per parameter (summed over G)    | `C_param [P]`          |
| units (neurons / channels)     | per unit × per old group         | `C_unit [U, G]`        |
| layers                         | per layer × per old group        | `C_layer [L, G]`       |

The parameter-path term of step `t` is integrated with the trapezoidal rule
(`euler` and `simpson` available for ablation):

    ΔL_c^(t) ≈ ½ (∇L_c(θ_t) + ∇L_c(θ_{t+1})) · Δθ_t .

For SGD without momentum, `Δθ_t = −η ∇ Σ_i ℓ_i(θ_t)`, where `ℓ_i` are the
*weighted per-sample loss terms* whose sum is the batch objective (new samples,
replay samples, regulariser). Hence the step contribution splits **exactly** over
sources: `C_i = −η ∇ℓ_i · ḡ_c`. Per-sample inner products are computed with a
double-backward identity, which is exact also under BatchNorm in train mode
(the batch coupling is part of `∇ℓ_i`). A-GEM's projection is linear given the
conflict decision and is folded into the per-sample terms.

Completeness error (reported everywhere):

    ε_c = |Σ sources − ΔL_c| / |ΔL_c| ,   ε_tot = |Σ_c Σ sources − Σ_c ΔL_c| / Σ_c |ΔL_c| .

---------------------------------------------------------------------------

## 1. Research questions and hypotheses

| ID  | Question | Hypothesis (pre-stated) | Primary metric |
|-----|----------|-------------------------|----------------|
| RQ1 | Is the ledger complete, and are single-point explanations adequate? | H1a: ε_tot < 5 % (GN / no-norm) and < 10 % (BN) at standard step sizes. H1b: first-order explanations at θ_start or θ_end deviate from ΔL by > 50 % or flip sign. | ε_tot; ratio (single-point / ΔL) |
| RQ2 | Are data attributions counterfactually faithful? | H2: removing the top-q % harmful samples and retraining reduces forgetting more than every baseline except TracIn-CP, at matched new-task cost; LDS(ledger) ≥ LDS(baselines). | Forgetting reduction ρ_q; LDS (Spearman) |
| RQ3 | Are attributions class-specific ("surgical")? | H3: removal targeted at old group c* improves c* ≥ 5× more than it improves other groups. | Specificity S = gain(c*) / mean |gain(others)| |
| RQ4 | Where does forgetting live (parameters / units / layers), and is it separable from learning? | H4a: unit-level ledger scores are more faithful under *freeze-and-retrain* than under *rollback*. H4b: an entanglement index (rank-corr. of forgetting vs learning credit per unit) predicts when parameter-level repair is impossible. | Recovered-forgetting fraction vs new-task cost (Pareto AUC); entanglement ι |
| RQ5 | How do CL methods prevent forgetting, mechanistically? | H5: the ledger decomposes the forgetting *prevented* by ER / DER++ / EWC / A-GEM into identifiable sources (replay slots, regulariser, projection), and these account for ≥ 90 % of the difference to fine-tuning. | Source shares; residual |
| RQ6 | Are interference maps semantically meaningful? | H6: the new-class × old-class interference map correlates with feature-space class similarity (Spearman > 0.3), and exhibits sparse, human-readable structure. | Spearman; Gini sparsity |
| RQ7 | Can forgetting be forecast early? | H7: ledger scores after 10/20/50 % of the task correlate with final scores (ρ ≥ 0.5 at 20 %) and retain ≥ 50 % of the removal benefit. | Spearman; ρ_q (early) / ρ_q (full) |
| RQ8 | What does it cost, and how sensitive is it? | H8: overhead ≤ (G+3)× a plain step; conclusions stable to probe size ≥ 20/class, lr, batch size, norm type. | wall-clock, memory; rank stability |

Negative outcomes are reported, not dropped. For each RQ the results section has
a pre-written sentence for the "hypothesis rejected" branch.

---------------------------------------------------------------------------

## 2. Benchmarks

| ID    | Scenario | Tasks | Model | Epochs/task | Batch | η | Norm | Where |
|-------|----------|-------|-------|-------------|-------|---|------|-------|
| PM5   | Permuted-MNIST, domain-IL | 5 | MLP 784-256-256-10 | 3 | 32 | 0.05 | – | CPU |
| SF2   | Split-FashionMNIST, domain-IL (pilot) | 2 | MLP | 3 | 32 | 0.05 | – | CPU |
| SM5   | Split-MNIST, class-IL & task-IL | 5 | MLP | 3 | 32 | 0.05 | – | CPU |
| SC10  | Seq-CIFAR-10, class-IL & task-IL | 5 | reduced ResNet-18 (nf=20) | 10 | 32 | 0.03 | GN (main) / BN (abl.) | Colab/Kaggle |
| SC100 | Seq-CIFAR-100, class-IL & task-IL | 10 | reduced ResNet-18 (nf=20) | 10 | 32 | 0.03 | GN / BN | Colab/Kaggle |
| (LLM) | continual instruction tuning, LoRA | 3–4 | Pythia-410M / Qwen-0.5B | – | – | – | – | phase 2 |

Probe set: 20 held-out training images per class (never trained on), for
every method identical. All methods (ledger and baselines) get the same probe
set as their only access to old data, except replay methods, which also have
their buffer (identical across attribution methods).

Groups `G`: per old class (default); per old task for SC100 (`group_by: task`),
per-class maps computed for the last task only.

## 3. Continual learners explained

`finetune` (SGD), `er` (reservoir-equivalent bottom-k hash buffer, M = 500),
`derpp` (α = 0.1, β = 0.5, M = 500), `ewc` (online, λ = 100 MNIST / 10 CIFAR,
γ = 1), `agem` (M = 500, ref batch 64). Hyper-parameters follow Mammoth
defaults where they exist and are fixed before running.

The hash buffer keeps the M samples with the smallest `hash(seed, task, idx)`
seen so far; it is a uniform sample of the stream and is **stable under removal
of other samples**, which keeps counterfactual retraining clean.

## 4. Baselines

Data-level (per new sample, per old group):
`random`; `loss@θ_start`; `grad-dot@θ_start` (static TracIn); `grad-cos@θ_start`;
`TracIn-CP-m` for m ∈ {3, 10} evenly spaced checkpoints of the task;
`feature-proximity` (max cosine of the sample's penultimate features to old
class centroids at θ_start); `ledger-early-{10,20,50}`.

Parameter/unit-level: `random`; `|Δθ|`; `Fisher·Δθ²` (Fisher on the probe set at
θ_start, i.e. the EWC penalty); `Taylor@θ_end` (∇L·Δθ at θ_end);
`ledger`; `ledger-net` (forgetting credit − λ·learning credit, λ normalising the
totals).

## 5. Interventions (ground truth)

* **Removal-and-retrain** (RQ2, RQ3, RQ7): retrain task k from the saved
  start-of-task state (parameters, buffer, regulariser state, RNG) with the
  selected samples removed and **the same data order and per-sample
  augmentation** for the remaining samples (augmentation randomness is a hash of
  `(seed, task, epoch, sample index)`). q ∈ {1, 5, 10, 20} %.
* **Linear Datamodeling Score** (RQ2): M = 32 (CPU) / 16 (GPU) random 50 %
  subsets; Spearman between predicted Σ_{i∈S} C_i and realised ΔL_c(S),
  averaged over groups.
* **Rollback** (RQ4): reset the top-k % parameters / units / layers to θ_start.
* **Freeze-and-retrain** (RQ4): retrain task k with the top-k % units frozen.
  k ∈ {0.5, 1, 2, 5, 10} %.

## 6. Metrics

Average accuracy A_K, average forgetting F_K (Chaudhry et al. 2018), per-class
accuracy matrix; ε (completeness); ρ_q = (F_full − F_removed) / F_full on the old
groups; new-task cost Δacc_new; Pareto AUC of (recovered forgetting, retained
new-task accuracy) over k; LDS; specificity S; entanglement ι; overhead.

## 7. Statistics

5 seeds per configuration (seed controls data split, permutations, init,
order, augmentation). Report mean ± 95 % t-interval. Method comparisons are
paired over seeds (same seed ⇒ same model, same data); we report paired
t-tests and Wilcoxon signed-rank tests, Holm–Bonferroni corrected within each
table. Effect sizes (Cohen's d_z) are reported alongside p-values.

## 8. Checkpointing and reproducibility

* Everything is resumable: `state/latest.pt` (atomic write) holds model, learner
  state, ledger accumulators, RNG states and the exact position
  `(task, epoch, batch)`; it is written every `ckpt_every` steps and at every
  epoch end. Re-running the same command resumes.
* Per task `k` we keep: `snapshots/task{k}_start.pt` and `task{k}_end.pt` (full
  learner state), `m` intra-task parameter checkpoints for TracIn-CP, and the
  ledger tensors `ledger/task{k}.pt`.
* Every run writes `config.json` (resolved), `env.json` (versions, GPU),
  `git.txt` (commit hash) and `metrics.jsonl`.

## 9. Compute budget (estimates)

| benchmark | tracked run / seed | interventions / seed | where |
|-----------|-------------------|----------------------|-------|
| PM5, SM5, SF2 | 5–15 min | 20–60 min | CPU (2 cores) |
| SC10 | ≈ 45 min (T4) | ≈ 2–3 h (T4) | Colab / Kaggle |
| SC100 | ≈ 2 h (T4, task groups) | ≈ 4–6 h (T4) | Kaggle (P100/T4×2) |

## Deviations

*(none yet)*
