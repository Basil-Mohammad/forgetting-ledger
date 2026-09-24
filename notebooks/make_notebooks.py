"""Generates the Colab and Kaggle notebooks (run: python notebooks/make_notebooks.py).

Two experiments:
  cifar : reduced ResNet-18 + BatchNorm on Split CIFAR-10 (10 seeds)
  llm   : Pythia-160M + LoRA, AG News -> DBpedia (10 seeds), plus the re-tracked Split CIFAR-10 CNN with
          16 quadrature sub-intervals (completeness check, 10 short runs)
Kaggle "GPU T4 x2" sessions run two jobs in parallel (one per GPU).
"""
import nbformat as nbf

REPO = "Basil-Mohammad/forgetting-ledger"
URL = "https://github.com/Basil-Mohammad/forgetting-ledger.git"

EXPERIMENTS = {
    "cifar": dict(
        title="ResNet-18 + BatchNorm on Split CIFAR-10",
        jobs='JOBS = [f"all configs/c10_resnet_bn.yaml seed={s}" for s in range(10)]',
        hours="about 36 min per seed on a T4",
        smoke=["all", "configs/c10_resnet_bn.yaml", "seed=99", "train_per_task=320", "first_task_train=640",
               "train.first_task_epochs=1", "train.epochs=1", "probe_per_class=4", "ledger.tracin_checkpoints=3",
               "interventions.reps=1", "interventions.removal_fracs=[0.1]", "interventions.lds_subsets=2",
               "interventions.surgery_targets=1", "interventions.param_fracs=[0.1]"],
        pip="",
    ),
    "llm": dict(
        title="Language model (Pythia-160M + LoRA, AG News -> DBpedia) and CIFAR-10 CNN completeness",
        jobs=('JOBS = [f"all configs/llm_pythia.yaml seed={s}" for s in range(10)]\n'
              'JOBS += [f"train configs/c10_cnn.yaml seed={s} ledger.max_intervals=16 tag=n16" for s in range(10)]'),
        hours="about 50-60 min per language-model seed and 5-10 min per CNN run on a T4",
        smoke=["all", "configs/llm_pythia.yaml", "seed=99", "first_task_train_per_class=40", "train_per_class=24",
               "test_per_class=20", "probe_per_class=4", "ledger.tracin_checkpoints=3", "interventions.reps=1",
               "interventions.removal_fracs=[0.1]", "interventions.lds_subsets=2", "interventions.surgery_targets=1"],
        pip="transformers datasets",
    ),
}

SETUP_REPO = '''import os, subprocess
{token}
url = f"https://{{tok}}@github.com/{repo}.git" if tok else "{url}"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", url, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
subprocess.run(["pip", "-q", "install", "-r", f"{{REPO_DIR}}/requirements.txt"{extra}], check=True)
import torch
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or "NO GPU")'''

SMOKE = '''# Smoke test on this GPU: the whole pipeline at a tiny scale (a few minutes). Run once before the real jobs.
import subprocess, sys, time
t0 = time.time()
subprocess.run([sys.executable, "scripts/run.py", *{smoke}, f"out_root={{RUN_ROOT}}/../_smoke", f"data_root={{DATA_ROOT}}"],
               cwd=REPO_DIR, check=True)
print(f"smoke test OK in {{(time.time() - t0) / 60:.1f}} min")'''

JOBS = '''# One entry per job: "<command> <config> [overrides...]". Every job is resumable; finished steps are skipped.
{jobs}
JOBS = [j.split() for j in JOBS]
print(len(JOBS), "jobs")'''

RUNNER = '''# Runs the jobs, one per GPU in parallel (Kaggle "T4 x2": two at a time). Re-running this cell after a
# disconnect resumes every job from its last checkpoint. Per-job logs: <RUN_ROOT>/../logs/.
import subprocess, sys, time, os, threading, queue, torch
LOGS = f"{{RUN_ROOT}}/../logs"; os.makedirs(LOGS, exist_ok=True)
NGPU = max(1, torch.cuda.device_count())
q = queue.Queue()
for j in JOBS:
    q.put(j)
T_START = time.time()
def worker(gpu):
    while True:
        if (time.time() - T_START) / 3600 > LIMIT_H:
            print(f"[gpu{{gpu}}] time budget reached - save and continue in a new session", flush=True); return
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        name = "_".join(job[1:]).replace("/", "_").replace("=", "-")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        t0 = time.time()
        print(f"[gpu{{gpu}}] >>> {{' '.join(job)}}", flush=True)
        with open(f"{{LOGS}}/{{name}}.log", "a") as f:
            rc = subprocess.call([sys.executable, "scripts/run.py", *job, f"out_root={{RUN_ROOT}}", f"data_root={{DATA_ROOT}}"],
                                 cwd=REPO_DIR, stdout=f, stderr=subprocess.STDOUT, env=env)
        print(f"[gpu{{gpu}}] <<< exit {{rc}} after {{(time.time() - t0) / 60:.1f}} min: {{' '.join(job)}}", flush=True)
        if rc != 0:
            print(open(f"{{LOGS}}/{{name}}.log").read()[-3000:], flush=True)
threads = [threading.Thread(target=worker, args=(g,)) for g in range(NGPU)]
[t.start() for t in threads]; [t.join() for t in threads]
print("all workers finished")'''

PROGRESS = '''# Progress overview: forgetting and ledger completeness of every finished tracked run.
import json, glob, os
for d in sorted(glob.glob(f"{RUN_ROOT}/*-s*")):
    f = f"{d}/metrics.jsonl"
    m = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    t1 = [x for x in m if x.get("task") == 1]
    steps = sorted(os.listdir(f"{d}/interventions")) if os.path.isdir(f"{d}/interventions") else []
    if t1:
        a0 = [x for x in m if x.get("task") == 0][0]["task_acc"][0]
        print(os.path.basename(d), f"| forgetting {100 * (a0 - t1[0]['task_acc'][0]):.1f} pp",
              f"| completeness {100 * (t1[0]['completeness'] or 0):.2f} %", "| interventions:", [s.split('_')[0] for s in steps])
    else:
        print(os.path.basename(d), "| tracked run not finished yet")'''

EXPORT = '''# Compact results for the paper (drops model snapshots, resumable states and the per-parameter arrays that
# the paper analysis does not need). Download the printed zip file and send it back.
import os, glob, shutil, subprocess, torch
DST = "/tmp/flgr_slim"
DROP = {{"param", "learn_param", "unit_of", "layer_of"}}
shutil.rmtree(DST, ignore_errors=True)
for run in sorted(glob.glob(f"{{RUN_ROOT}}/*")):
    for root, dirs, files in os.walk(run):
        dirs[:] = [d for d in dirs if d not in ("snapshots", "state", "adam_tmp")]
        for f in files:
            src = os.path.join(root, f); dst = os.path.join(DST, os.path.relpath(src, RUN_ROOT))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if f.endswith(".pt") and "/ledger/" in src:
                d = torch.load(src, map_location="cpu", weights_only=False)
                torch.save({{k: v for k, v in d.items() if k not in DROP}}, dst)
            else:
                shutil.copy2(src, dst)
out = f"{{RUN_ROOT}}/../flgr_results_{name}.zip"
if os.path.exists(out):
    os.remove(out)
subprocess.run(f"cd {{DST}} && zip -qr {{out}} .", shell=True, check=True)
print(out, round(os.path.getsize(out) / 1e6, 1), "MB")'''


def _cells(exp, platform):
    E = EXPERIMENTS[exp]
    extra = "".join(f', "{p}"' for p in E["pip"].split())
    c = []
    if platform == "colab":
        c.append(nbf.v4.new_markdown_cell(
            f"# Forgetting Ledger — {E['title']} (Google Colab)\n\n"
            "1. *Runtime → Change runtime type → T4 GPU*.\n"
            f"2. Only if the repository is private: add a Colab secret **`GH_TOKEN`** (key icon) that can read `{REPO}`.\n"
            "3. Run the cells in order. Outputs and checkpoints live on Google Drive: after a disconnect, re-run all "
            "cells (you may skip the smoke test) and every job resumes from its last checkpoint.\n"
            f"4. Expected time: {E['hours']}. When everything is finished run the last cell and download "
            f"`flgr_results_{exp}.zip` from `MyDrive/forgetting-ledger/`."))
        c.append(nbf.v4.new_code_cell(
            "from google.colab import drive, userdata\n"
            "drive.mount('/content/drive')\n"
            "BASE = '/content/drive/MyDrive/forgetting-ledger'\n"
            "RUN_ROOT, DATA_ROOT, REPO_DIR = f'{BASE}/runs', '/content/data', '/content/forgetting-ledger'\n"
            "LIMIT_H = 1000\n"
            "import os; os.makedirs(RUN_ROOT, exist_ok=True)"))
        token = "try:\n    tok = userdata.get('GH_TOKEN')\nexcept Exception:\n    tok = None   # public repository: no token needed"
    else:
        c.append(nbf.v4.new_markdown_cell(
            f"# Forgetting Ledger — {E['title']} (Kaggle)\n\n"
            "1. *Session options → Accelerator → **GPU T4 x2*** (two jobs run in parallel), *Internet → On*.\n"
            "2. Only if the repository is private: *Add-ons → Secrets* → **`GH_TOKEN`**.\n"
            "3. Use **Save Version → Save & Run All**: the notebook then runs in the background (up to 12 h) and the "
            "last cell writes the results file to the version's *Output*.\n"
            "4. To continue in a new session, add the previous version's output as an input: the first cell copies "
            "`runs/` back and every job resumes.\n"
            f"5. Expected time: {E['hours']}."))
        c.append(nbf.v4.new_code_cell(
            "import os, glob, shutil\n"
            "RUN_ROOT, DATA_ROOT, REPO_DIR = '/kaggle/working/runs', '/kaggle/tmp/data', '/kaggle/tmp/forgetting-ledger'\n"
            "LIMIT_H = 11.0   # Kaggle's hard limit is 12 h: no new job is started after 11 h\n"
            "os.makedirs(RUN_ROOT, exist_ok=True)\n"
            "for prev in glob.glob('/kaggle/input/*/runs'):\n"
            "    print('restoring', prev)\n"
            "    shutil.copytree(prev, RUN_ROOT, dirs_exist_ok=True)"))
        token = ("try:\n    from kaggle_secrets import UserSecretsClient\n    tok = UserSecretsClient().get_secret('GH_TOKEN')\n"
                 "except Exception:\n    tok = None   # public repository: no token needed")
    c.append(nbf.v4.new_code_cell(SETUP_REPO.format(token=token, repo=REPO, url=URL, extra=extra)))
    c.append(nbf.v4.new_code_cell(SMOKE.format(smoke=repr(E["smoke"]))))
    c.append(nbf.v4.new_code_cell(JOBS.format(jobs=E["jobs"])))
    c.append(nbf.v4.new_code_cell(RUNNER.format()))
    c.append(nbf.v4.new_code_cell(PROGRESS))
    c.append(nbf.v4.new_code_cell(EXPORT.format(name=exp)))
    nb = nbf.v4.new_notebook()
    nb["cells"] = c
    nb["metadata"] = ({"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
                      if platform == "colab" else {"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    return nb


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for exp in EXPERIMENTS:
        for platform in ("colab", "kaggle"):
            nbf.write(_cells(exp, platform), os.path.join(here, f"{platform}_{exp}.ipynb"))
    print("written")
