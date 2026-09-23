"""Generates the Colab and Kaggle notebooks (run: python notebooks/make_notebooks.py)."""
import nbformat as nbf

REPO = "Basil-Mohammad/forgetting-ledger"

JOBS = '''# One line per job: <command> <config> [overrides...]
# `all` = train -> scores -> removal -> surgery -> lds -> params (each step resumable).
JOBS = """
all configs/sc10_resnet.yaml seed=0
all configs/sc10_resnet.yaml seed=1
all configs/sc10_resnet.yaml seed=2
all configs/sc10_resnet.yaml seed=3
all configs/sc10_resnet.yaml seed=4
train configs/sc10_resnet.yaml seed=0 learner.name=er
train configs/sc10_resnet.yaml seed=0 learner.name=derpp
train configs/sc10_resnet.yaml seed=0 learner.name=ewc
train configs/sc10_resnet.yaml seed=0 learner.name=agem
train configs/sc10_resnet.yaml seed=0 model.norm=bn
all configs/sc10_resnet.yaml seed=0 setting=task
"""
JOBS = [l.split() for l in JOBS.strip().splitlines() if l.strip() and not l.startswith("#")]
print(len(JOBS), "jobs")'''

RUNNER = '''import subprocess, sys, time, os
def run_job(job):
    cmd = [sys.executable, "scripts/run.py", *job, f"out_root={RUN_ROOT}", f"data_root={DATA_ROOT}"]
    print(">>>", " ".join(job), flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    print(f"<<< exit {p.returncode} after {(time.time()-t0)/60:.1f} min", flush=True)
    return p.returncode

# Re-running this cell after a disconnect resumes exactly where it stopped:
# finished steps are skipped and the current step restarts from state/latest.pt.
for job in JOBS:
    if run_job(job) != 0:
        print("job failed - fix and re-run this cell (it will resume)"); break'''

SMOKE = '''# Smoke test + timing on this GPU (about 1-2 minutes). Run once before the real jobs.
import subprocess, sys
subprocess.run([sys.executable, "scripts/run.py", "train", "configs/sc10_resnet.yaml", "seed=99", "n_tasks=2",
                "train_per_task=640", "train.epochs=1", f"out_root={RUN_ROOT}/_smoke", f"data_root={DATA_ROOT}"],
               cwd=REPO_DIR, check=True)'''

AGG = '''import subprocess, sys
subprocess.run([sys.executable, "-m", "flgr.analysis.aggregate", "--runs", RUN_ROOT, "--out", f"{RUN_ROOT}/../results"],
               cwd=REPO_DIR, check=True)
from IPython.display import Markdown, display
display(Markdown(open(f"{RUN_ROOT}/../results/results.md").read()))'''

ZIP = '''# Compact archive of everything needed for the paper (ledger tensors, scores, interventions, evals;
# model snapshots are excluded to keep it small).
import shutil, os, subprocess
out = f"{RUN_ROOT}/../flgr_results"
subprocess.run(f"cd {RUN_ROOT}/.. && zip -qr flgr_results.zip results runs -x '*/snapshots/*' '*/state/*'", shell=True)
print(os.path.getsize(f"{RUN_ROOT}/../flgr_results.zip")/1e6, "MB")'''


def colab():
    nb = nbf.v4.new_notebook()
    c = []
    c.append(nbf.v4.new_markdown_cell(
        "# Forgetting Ledger — CIFAR experiments (Google Colab)\n\n"
        "1. *Runtime → Change runtime type → GPU* (T4 is enough).\n"
        "2. Add a Colab secret **`GH_TOKEN`** (key icon on the left) holding a GitHub token with read access to the "
        f"private repository `{REPO}`.\n"
        "3. Run all cells. All outputs and checkpoints live on Google Drive, so after a disconnect simply "
        "**re-run all cells** — every job resumes from its last checkpoint."))
    c.append(nbf.v4.new_code_cell(
        "from google.colab import drive, userdata\n"
        "drive.mount('/content/drive')\n"
        "BASE = '/content/drive/MyDrive/forgetting-ledger'\n"
        "RUN_ROOT, DATA_ROOT, REPO_DIR = f'{BASE}/runs', '/content/data', '/content/forgetting-ledger'\n"
        "import os; os.makedirs(RUN_ROOT, exist_ok=True)"))
    c.append(nbf.v4.new_code_cell(
        "import os, subprocess\n"
        "tok = userdata.get('GH_TOKEN')\n"
        "if not os.path.exists(REPO_DIR):\n"
        f"    subprocess.run(['git', 'clone', f'https://{{tok}}@github.com/{REPO}.git', REPO_DIR], check=True)\n"
        "else:\n"
        "    subprocess.run(['git', '-C', REPO_DIR, 'pull'], check=True)\n"
        "subprocess.run(['pip', '-q', 'install', '-r', f'{REPO_DIR}/requirements.txt'], check=True)\n"
        "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"))
    c.append(nbf.v4.new_code_cell(SMOKE))
    c.append(nbf.v4.new_code_cell(JOBS))
    c.append(nbf.v4.new_code_cell(RUNNER))
    c.append(nbf.v4.new_code_cell(AGG))
    c.append(nbf.v4.new_code_cell(ZIP))
    nb["cells"] = c
    nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    return nb


def kaggle():
    nb = nbf.v4.new_notebook()
    c = []
    c.append(nbf.v4.new_markdown_cell(
        "# Forgetting Ledger — CIFAR experiments (Kaggle)\n\n"
        "1. *Settings → Accelerator → GPU T4 x2 or P100*, *Internet → On*.\n"
        "2. *Add-ons → Secrets*: add **`GH_TOKEN`** (GitHub token with read access to the private repository).\n"
        "3. Kaggle sessions stop after 12 h and only `/kaggle/working` is kept when you **Save Version (Save & Run All)**. "
        "To continue in a new session, add the previous version's output as an input dataset: the first cell copies "
        "`runs/` back and every job resumes from its checkpoint."))
    c.append(nbf.v4.new_code_cell(
        "import os, glob, shutil\n"
        "RUN_ROOT, DATA_ROOT, REPO_DIR = '/kaggle/working/runs', '/kaggle/tmp/data', '/kaggle/tmp/forgetting-ledger'\n"
        "os.makedirs(RUN_ROOT, exist_ok=True)\n"
        "# resume from a previous version's output attached as input\n"
        "for prev in glob.glob('/kaggle/input/*/runs'):\n"
        "    print('restoring', prev)\n"
        "    shutil.copytree(prev, RUN_ROOT, dirs_exist_ok=True)"))
    c.append(nbf.v4.new_code_cell(
        "import subprocess\n"
        "from kaggle_secrets import UserSecretsClient\n"
        "tok = UserSecretsClient().get_secret('GH_TOKEN')\n"
        "if not os.path.exists(REPO_DIR):\n"
        f"    subprocess.run(['git', 'clone', f'https://{{tok}}@github.com/{REPO}.git', REPO_DIR], check=True)\n"
        "subprocess.run(['pip', '-q', 'install', '-r', f'{REPO_DIR}/requirements.txt'], check=True)\n"
        "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"))
    c.append(nbf.v4.new_code_cell(SMOKE))
    c.append(nbf.v4.new_code_cell(JOBS.replace("configs/sc10_resnet.yaml seed=0 setting=task", "configs/sc10_resnet.yaml seed=0 setting=task\nall configs/sc100_resnet.yaml seed=0\nall configs/sc100_resnet.yaml seed=1\nall configs/sc100_resnet.yaml seed=2")))
    c.append(nbf.v4.new_code_cell(
        "# Kaggle hard limit is 12 h: stop starting new jobs after 11 h so the version can be saved cleanly.\n"
        "import time; T_START = time.time(); LIMIT_H = 11.0\n" + RUNNER.replace(
            "for job in JOBS:\n", "for job in JOBS:\n    if (time.time() - T_START) / 3600 > LIMIT_H:\n        print('time budget reached - Save Version and continue in a new session'); break\n")))
    c.append(nbf.v4.new_code_cell(AGG))
    c.append(nbf.v4.new_code_cell(ZIP))
    nb["cells"] = c
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    return nb


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    nbf.write(colab(), os.path.join(here, "colab_cifar.ipynb"))
    nbf.write(kaggle(), os.path.join(here, "kaggle_cifar.ipynb"))
    print("written")
