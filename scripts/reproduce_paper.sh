#!/bin/bash
# Every run behind the paper (CPU, OMP_NUM_THREADS=1 per process; runs are resumable).
# Usage: bash scripts/reproduce_paper.sh [RUNS_DIR] [DATA_DIR]
set -e
R=${1:-./runs}; D=${2:-./data}
export OMP_NUM_THREADS=1
run() { python scripts/run.py "$@" data_root=$D out_root=$R; }

# core benchmarks (10 seeds): tracked run, scores, removal, LDS, surgery, freeze/rollback
for s in 0 1 2 3 4 5 6 7 8 9; do run all configs/pm2_mlp.yaml seed=$s; run all configs/sf2_mlp.yaml seed=$s; run all configs/c10_cnn.yaml seed=$s; done

# transfer targets (wide MLP, CNN on the same data; 5 seeds)
for s in 0 1 2 3 4; do for b in pm2 sf2; do
  run train configs/${b}_mlp.yaml seed=$s model.hidden=400 model.depth=3 tag=wide
  run train configs/${b}_mlp.yaml seed=$s model.name=cnn model.width=16 ledger.rule=trapezoid ledger.also_euler=false tag=cnn
done; done

# dose-response in task overlap (5 levels x 10 seeds) and Adam (10 seeds)
for s in 0 1 2 3 4 5 6 7 8 9; do
  for f in 0.25 0.5 0.7 0.85 1.0; do run all configs/ppm_dose.yaml seed=$s perm_frac=$f tag=pf$f; done
  run all configs/ppm_dose.yaml seed=$s perm_frac=1.0 train.optimizer=adam train.lr=0.001 tag=adam
done

# five continual learners on Split MNIST (class-IL, 3 tasks, 10 seeds)
for s in 0 1 2 3 4 5 6 7 8 9; do for l in finetune er derpp ewc agem; do run train configs/sm3_mlp.yaml seed=$s learner.name=$l; done; done

# numerical ablations (3 seeds): integration rule and learning rate
for s in 0 1 2; do
  for r in euler trapezoid simpson adaptive; do run train configs/pm2_mlp.yaml seed=$s train_per_task=5000 ledger.rule=$r ledger.also_euler=false tag=rule-$r; done
  for lr in 0.025 0.05 0.2; do run train configs/pm2_mlp.yaml seed=$s train_per_task=5000 train.lr=$lr ledger.also_euler=false tag=lr$lr; done
done

# proxy curation (two-fold cross-fitting over seeds; MLP proxies -> other architectures)
C="python scripts/curation.py --runs $R --reps 3"
for base in pmnist-domain-mlp-finetune sfmnist-task-mlp-finetune; do
  $C --q 0.05 0.1 0.2 --proxy $base-s{0..4} --targets $base-s{5..9}
  $C --q 0.05 0.1 0.2 --proxy $base-s{5..9} --targets $base-s{0..4}
done
b=scifar10-task-cnn-none-finetune
$C --q 0.1 --proxy $b-s{0..4} --targets $b-s{5..9}
$C --q 0.1 --proxy $b-s{5..9} --targets $b-s{0..4}
for p in pmnist-domain sfmnist-task; do
  $C --q 0.05 0.1 0.2 --proxy $p-mlp-finetune-s{5..9} --targets $(for s in 0 1 2 3 4; do echo -n "$p-mlp-finetune-wide-s$s $p-cnn-finetune-cnn-s$s "; done)
done

# every figure (PDF) and table / macro (LaTeX) of the paper
python -m flgr.analysis.paper --runs $R --fig ./paper/figures --tab ./paper/generated
