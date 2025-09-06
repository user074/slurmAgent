# slurmAgent

Make SLURM feel like `python script.py`.

**One command:**

```bash
slurm python train.py --lr 3e-4
```

slurmAgent will (1) guess resources (optionally with Gemini), (2) render a batch script, (3) `sbatch` submit, (4) live-tail logs & poll status, and (5) triage common errors.

---

## Features (MVP)

* **One-liner UX**: `slurm <your command>` (no need to hand-craft `#SBATCH`).
* **AI resource guessing (optional)** via Gemini CLI; deterministic heuristics as fallback.
* **Templated batch scripts** (Jinja2).
* **Live monitoring**: polls `squeue`, tails stdout.
* **Basic triage**: detects OOM, TIME\_LIMIT, GRES/account/QOS issues and suggests fixes.
* **Dry-run**: preview the exact script before submitting.

---

## Requirements

* Python 3.8+ and SLURM CLI tools on your PATH: `sbatch`, `squeue`, `sacct` (recommended).
* Optional: Google Gemini CLI for AI resource guesses.

Install deps:
Install 'npm' in conda:
```bash
conda install -c conda-forge nodejs
```

```bash
npm install -g @google/gemini-cli     # optional, for --ai
pip install pyyaml jinja2
```

Authenticate Gemini once:

```bash
gemini   # follow the login prompt
```

---

## Getting the `slurm` command

### Option A (demo, quickest): shell alias

```bash
alias slurm='python /ABSOLUTE/PATH/TO/slurm.py'
# add the line to ~/.zshrc or ~/.bashrc to make it permanent
```

### Option B (clean): put the script on your PATH

```bash
mv slurm.py slurm
chmod +x slurm
mkdir -p ~/.local/bin
mv slurm ~/.local/bin/
# keep the template next to it:
mkdir -p ~/.local/bin/templates
cp templates/job.sh.j2 ~/.local/bin/templates/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc  # or ~/.bashrc
source ~/.zshrc
```

---

## Quick start
You don't need to know anything about your cluster, partition, account, etc.

1. **Generate a config** (`slurm.yaml` in your project root):

```bash
# one-time setup
slurm --init         # writes slurm.yaml + submits a small probe job
slurm --doctor       # optional: prints cluster/accounts/partitions health

# run something
slurm --ai --explain python train.py --lr 3e-4
# or override per-run:
slurm --partition gpu --gpus 2 --mem 64G python train.py
```

### What `--init` does

* Checks SLURM tools (`sbatch`, `squeue`, `sacct`, `sinfo`).
* Detects your **cluster name**, available **accounts**, and **partitions**.
* Picks a usable partition (GPU if available) and sets `defaults.gres: gpu:1` when appropriate.
* Writes `slurm.yaml` with safe defaults:

  * logs under `runs/{date}/{job_name}/out_%j.log` / `err_%j.log`
  * modest CPU/MEM/TIME, editable later
* Submits a tiny **probe** job and prints:

  * compute-node hostname, shared path check, Python/conda info, `nvidia-smi` (if GPUs)

### When you might still edit `slurm.yaml`

* Your site **requires an account/QOS** and `--init` couldn’t discover it.
* You want specific **modules/conda env** or a different default **partition**.
* Paths: if probe shows login vs compute paths differ, point `workdir` (i.e., where you run `slurm` from) to a shared FS (e.g., `$HOME`/`$SCRATCH`).

---

## Manual config (advanced / optional)

If you prefer, keep a custom `slurm.yaml` in your repo and skip `--init`. All fields can still be overridden at run time via flags.

 **Prepare a manual config** (`slurm.yaml` in your project root):

```yaml
cluster: "my-hpc"
account: "ai_research"           # or leave empty if not required
default_partition: "gpu"         # set to a valid partition on your cluster
env:
  modules: ["cuda/12.1"]         # optional: environment modules to load
  conda_env: "base"              # optional: conda env to activate
logging:
  dir: "runs/{date}/{job_name}"
  stdout: "out_%j.log"
  stderr: "err_%j.log"
defaults:
  gpus: 1
  cpus_per_task: 8
  mem: "32G"
  time: "04:00:00"
  gres: "gpu:1"
  nodes: 1
  ntasks_per_node: 1
policy:
  auto_retry: false
```

2. **Run a dry-run (no submission):**

```bash
slurm --ai --dry-run --partition cpu python cpu_test.py
```

3. **Submit for real:**

```bash
slurm --ai python train.py --model llama-13b --batch_size 8
# or override anything explicitly:
slurm --gpus 2 --mem 64G --time 08:00:00 --partition a100 python train.py
```

---

## Usage

```bash
slurm [FLAGS] <your command and args>

Flags:
  --ai                 # ask Gemini CLI to guess resources
  --dry-run            # render script and print plan; do not submit
  --config slurm.yaml  # path to config (default: ./slurm.yaml)
  --partition NAME
  --gpus N
  --mem 64G
  --cpus 16
  --time HH:MM:SS
  --name myjob         # optional job name override
```

**Examples**

```bash
# CPU job on the cpu partition
slurm --partition cpu python scripts/eval_cpu.py

# GPU job with explicit overrides (no Gemini)
slurm --gpus 2 --mem 64G --cpus 16 --time 12:00:00 python train.py --epochs 50

# Dry-run preview
slurm --ai --dry-run python train.py --model llama-13b
```

---

## What gets created

* **Rendered script:** `runs/{date}/{job_name}/job.sh`
* **Logs:**

  * stdout → `runs/{date}/{job_name}/out_<jobid>.log`
  * stderr → `runs/{date}/{job_name}/err_<jobid>.log`
* **State:** `runs/{date}/{job_name}/state.json` (job id, cmd, timestamps)

The monitor prints `squeue` state and tails the last lines of stdout while the job runs. When it finishes, a brief `sacct` summary is shown.

---

## How it works (under the hood)

1. Parse your command and read `slurm.yaml` defaults.
2. **Resource inference**

   * If `--ai`: prompt Gemini CLI for `{gpus, mem, cpus, time, gres, partition}`.
   * Else: simple heuristics (e.g., detect large LLMs, batch size hints).
3. Render a Jinja2 template at `templates/job.sh.j2` with the chosen resources.
4. Submit with `sbatch`, capture the job id.
5. Poll `squeue` and tail logs; on exit, query `sacct`.
6. Classify common failures and print suggestions:

   * **OOM** → lower batch size / increase `--mem` or `--gpus`
   * **TIME\_LIMIT** → increase `--time` or enable checkpoints
   * **GRES/GPUs mismatch** → check `--gres` vs `--gpus`
   * **ACCOUNT/QOS** → set `account`/`partition` correctly

---

## Template (Jinja) quick look

`templates/job.sh.j2` (edit to match your cluster):

```jinja
#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --partition={{ partition }}
{% if account %}#SBATCH --account={{ account }}{% endif %}
#SBATCH --time={{ time }}
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks-per-node={{ ntasks_per_node }}
#SBATCH --cpus-per-task={{ cpus }}
#SBATCH --mem={{ mem }}
{% if gpus %}#SBATCH --gpus={{ gpus }}{% endif %}
{% if gres %}#SBATCH --gres={{ gres }}{% endif %}
#SBATCH --output={{ log_dir }}/{{ stdout }}
#SBATCH --error={{ log_dir }}/{{ stderr }}

set -euo pipefail
if command -v module >/dev/null 2>&1; then
  module purge
  {% for m in modules %}module load {{ m }}
  {% endfor %}
fi
source ~/.bashrc 2>/dev/null || true
conda activate {{ conda_env }} 2>/dev/null || true

cd {{ workdir }}
{{ command_line }}
```

---

## Tips & Troubleshooting

* **“Logs aren’t showing up”** → Ensure your working directory is on a shared filesystem visible to compute nodes (e.g., `$HOME` or `$SCRATCH`), not a login-only mount.
* **“GPU not found”** → Choose a GPU partition (e.g., `--partition gpu`) and ensure `defaults.gres: gpu:1` or `--gpus 1`.
* **“invalid gres / unknown gres”** → Some clusters allow **either** `--gpus` **or** `--gres`, not both. Set only one in config/flags.
* **“Invalid account / QOS”** → Set `account:` in `slurm.yaml` or use `--partition`/your site’s QOS correctly.
* **Modules vs Conda** → If your cluster doesn’t use Environment Modules, the template gracefully skips them.

---

## Roadmap

* `slurm status | logs <id> | cancel <id>` subcommands.
* Job arrays & dependency chains (sweep → train → eval).

---

## License

Apache 2.0

---

## Acknowledgements

* Built around SLURM CLI tools (`sbatch`, `squeue`, `sacct`).
* Optional AI resource guessing via \[Google Gemini CLI].
