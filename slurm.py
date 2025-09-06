#!/usr/bin/env python3
import argparse, os, subprocess, sys, time, json, re, datetime, pathlib
from pathlib import Path
import yaml
from jinja2 import Template
import shutil

def which(cmd): return shutil.which(cmd) is not None

def ensure_cmds(cmds):
    missing = [c for c in cmds if not which(c)]
    return missing

def get_cluster_name():
    try:
        out = sh("scontrol show config | sed -n 's/^ClusterName=\(.*\)$/\1/p' | head -n1", check=False)
        return out.strip() or "default"
    except Exception:
        return "default"

def detect_accounts():
    # returns list of accounts (may be empty)
    try:
        out = sh("sacctmgr -Pn show assoc user=$USER format=Account", check=False)
        accs = sorted(set([line.strip() for line in out.splitlines() if line.strip() and line.strip() != 'Account']))
        return accs
    except Exception:
        return []

def detect_partitions():
    # partition | avail | timelimit | cpus/node | mem/node | gres
    out = sh("sinfo -h -o '%P|%a|%l|%c|%m|%G'", check=False)
    parts = []
    for line in out.splitlines():
        if not line.strip(): continue
        P,a,l,c,m,G = (x.strip() for x in line.split("|",5))
        P = P.rstrip("*")  # remove default marker
        parts.append({"name": P, "avail": a, "time": l, "cpus": c, "mem": m, "gres": G})
    return parts

def pick_defaults_from_partitions(parts):
    # Prefer a GPU partition that is available, else first available partition
    gpu_like = [p for p in parts if p["avail"].lower().startswith("up") and ("gpu" in (p["gres"] or "").lower() or "gpu" in p["name"].lower())]
    non_gpu = [p for p in parts if p["avail"].lower().startswith("up")]
    chosen = gpu_like[0] if gpu_like else (non_gpu[0] if non_gpu else (parts[0] if parts else None))
    if not chosen: return None, None
    # derive gres template
    gres = None
    G = (chosen["gres"] or "").lower()
    # common forms: "gpu:1", "gpu:a100:4", "gpu:tesla:v100:4"
    if "gpu" in G:
        # default to "gpu:1" if we see any GPU
        gres = "gpu:1"
    return chosen["name"], gres

def write_yaml_file(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def run_probe_job(partition, account, use_gres):
    # Create a tiny job that prints environment info from a compute node
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_dir = Path(f"runs/init-probe-{ts}")
    probe_dir.mkdir(parents=True, exist_ok=True)
    out_file = probe_dir / "probe_%j.out"
    err_file = probe_dir / "probe_%j.err"
    script = probe_dir / "probe.sh"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=slurm-probe",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time=00:03:00",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem=1G",
        f"#SBATCH --output={out_file}",
        f"#SBATCH --error={err_file}",
    ]
    if account: lines.append(f"#SBATCH --account={account}")
    if use_gres: lines.append(f"#SBATCH --gres={use_gres}")
    lines += [
        "set -euo pipefail",
        "echo '== probe =='",
        "echo USER: $(whoami)",
        "echo HOST: $(hostname)",
        "echo PWD_LOGIN_SIDE: " + os.getcwd(),
        "echo PWD_COMPUTE_SIDE: $(pwd -P)",
        "echo PYTHON: $(which python || true)",
        "python -V || true",
        "echo CONDA_PREFIX: ${CONDA_PREFIX:-}",
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available'",
        "echo MODULES:",
        "if command -v module >/dev/null 2>&1; then module list 2>&1 || true; else echo 'no environment modules'; fi",
        "echo DF_DOT:",
        "df -h . || true",
    ]
    script.write_text("\n".join(lines))
    os.chmod(script, 0o755)
    out = sh(f"sbatch --parsable {script}")
    job_id = out.split(";")[0].strip()
    # wait a short while for completion
    for _ in range(30):  # ~30*4s = 2min
        time.sleep(4)
        if not sh(f"squeue -j {job_id} -h -o '%T'", check=False):
            break
    # gather results
    out_path = str(out_file).replace("%j", job_id)
    err_path = str(err_file).replace("%j", job_id)
    out_txt = Path(out_path).read_text() if Path(out_path).exists() else ""
    err_txt = Path(err_path).read_text() if Path(err_path).exists() else ""
    return {"job_id": job_id, "dir": str(probe_dir.resolve()),
            "stdout": out_path, "stderr": err_path,
            "out": out_txt[-8000:], "err": err_txt[-8000:]}


ROOT = Path(__file__).resolve().parent

OOM_PATTERNS = [
    r"OutOfMemoryError", r"CUDA out of memory", r"OOM Killed", r"MemoryError"
]
TIME_PATTERNS = [r"TIME_LIMIT", r"Job .* cancelled at .* due to time limit"]
GRES_PATTERNS = [r"Invalid generic resource", r"invalid gres", r"Unknown gres"]
ACCOUNT_PATTERNS = [r"Invalid account", r"AssocGrp", r"QOSMax"]

def sh(cmd, check=True, capture=True):
    if capture:
        p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if check and p.returncode != 0:
            raise RuntimeError(f"cmd failed: {cmd}\n{p.stdout}")
        return p.stdout.strip()
    else:
        rc = subprocess.call(cmd, shell=True)
        if check and rc != 0:
            raise RuntimeError(f"cmd failed: {cmd}")
        return ""

def load_yaml(path):
    with open(path) as f: return yaml.safe_load(f)

def render_template(tmpl_path, ctx):
    with open(tmpl_path) as f:
        t = Template(f.read())
    return t.render(**ctx)

def guess_resources_with_rules(command, defaults):
    g = defaults.get("gpus", 1)
    mem = defaults.get("mem", "32G")
    cpus = defaults.get("cpus_per_task", 8)
    time_ = defaults.get("time", "04:00:00")
    gres = defaults.get("gres", f"gpu:{g}")
    reasons = []

    lower = command.lower()
    if any(k in lower for k in ["llama", "qwen2.5-14b", "mixtral", "deepseek"]):
        old_g, old_mem = g, mem
        g, mem = max(g, 2), "64G"
        gres = f"gpu:{g}"
        reasons.append(f"Detected large LLM keyword → bump GPUs {old_g}→{g}, mem {old_mem}→{mem}")
    if "--batch_size" in lower or "batch_size" in lower:
        if mem.endswith("32G"):
            reasons.append("Saw batch_size hint → bump mem 32G→48G")
            mem = "48G"
    if not reasons:
        reasons.append("Fallback to defaults (no heavy-model/batch hints found)")
    return dict(gpus=g, mem=mem, cpus=cpus, time=time_, gres=gres), reasons

def guess_resources_with_gemini(command, workdir, defaults):
    prompt = f"""
You are configuring a SLURM job for this user command:

```

{command}

```

Return ONLY a JSON object with two keys:
- "resources": {{gpus:int, mem:"32G", cpus:int, time:"HH:MM:SS", gres:"gpu:N" or null, partition:string or null}}
- "explain": {{ per-field short reasons, e.g., "gpus": "detected llama-13b" }}
"""
    try:
        out = sh(f"""gemini -m gemini-2.5-flash -p '''{prompt}'''""", capture=True)
        print(f"[gemini] {out}")
        m = re.search(r"\{.*\}", out, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        res = data.get("resources", data)
        exp = data.get("explain", {})
        r = {
            "gpus": int(res.get("gpus", defaults.get("gpus",1))),
            "mem": res.get("mem", defaults.get("mem","32G")),
            "cpus": int(res.get("cpus", defaults.get("cpus_per_task",8))),
            "time": res.get("time", defaults.get("time","04:00:00")),
            "gres": res.get("gres", defaults.get("gres", f"gpu:{{defaults.get('gpus',1)}}")),
            "partition": res.get("partition")
        }
        return r, exp
    except Exception as e:
        print(f"[warn] gemini guess failed, falling back to rules: {e}")
        return guess_resources_with_rules(command, defaults)

def classify_error(log_text):
    for p in OOM_PATTERNS:
        if re.search(p, log_text, re.IGNORECASE): return "OOM"
    for p in TIME_PATTERNS:
        if re.search(p, log_text, re.IGNORECASE): return "TIME_LIMIT"
    for p in GRES_PATTERNS:
        if re.search(p, log_text, re.IGNORECASE): return "GRES"
    for p in ACCOUNT_PATTERNS:
        if re.search(p, log_text, re.IGNORECASE): return "ACCOUNT/QOS"
    return None

def guess_job_name(cmd: str):
    toks = cmd.strip().split()
    # try to find a .py or -m module name
    for t in toks:
        if t.endswith(".py"):
            return Path(t).stem
    # else the first token (runner)
    return re.sub(r'[^A-Za-z0-9._-]+', "_", toks[0])[:40]

def main():
    ap = argparse.ArgumentParser(description="slurm MVP")
    ap.add_argument("runner", nargs=argparse.REMAINDER, help="your command, e.g., python train.py --lr 3e-4")
    ap.add_argument("--config", default="slurm.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ai", action="store_true", help="use Gemini CLI to guess resources")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--mem", default=None)
    ap.add_argument("--cpus", type=int, default=None)
    ap.add_argument("--time", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--init", action="store_true", help="generate slurm.yaml and run a compute-node probe")
    ap.add_argument("--doctor", action="store_true", help="check cluster/tools/logging and print guidance")
    args = ap.parse_args()

    if args.init or args.doctor:
        # 0) basic tooling
        missing = ensure_cmds(["sbatch", "squeue", "sacct", "sinfo"])
        if missing:
            print(f"[doctor] Missing required commands: {', '.join(missing)}")
            print("Install or load these via your HPC's environment modules.")
            if args.doctor and not args.init:
                sys.exit(1)

        # 1) cluster facts
        cluster = get_cluster_name()
        accounts = detect_accounts()
        parts = detect_partitions()
        part, gres = pick_defaults_from_partitions(parts)
        account = accounts[0] if accounts else None

        print("\n[doctor] cluster snapshot")
        print(json.dumps({
            "cluster": cluster,
            "accounts": accounts or ["<none found>"],
            "partitions": parts[:6] + ([{"...":"(truncated)"}] if len(parts)>6 else []),
            "chosen_partition": part,
            "suggested_gres": gres
        }, indent=2))

        if args.doctor and not args.init:
            # just a report
            sys.exit(0)

        # 2) synthesize minimal slurm.yaml
        cfg_path = Path(args.config)
        if cfg_path.exists():
            print(f"[init] {cfg_path} already exists; will not overwrite. (Delete it to regenerate.)")
        else:
            logging_dir = "runs/{date}/{job_name}"
            cfg = {
                "cluster": cluster,
                "account": account,
                "default_partition": part or "gpu",
                "env": {"modules": [], "conda_env": "base"},
                "logging": {"dir": logging_dir, "stdout": "out_%j.log", "stderr": "err_%j.log"},
                "defaults": {
                    "gpus": 1 if gres else 0,
                    "cpus_per_task": 4,
                    "mem": "16G" if not gres else "32G",
                    "time": "04:00:00",
                    "gres": gres,
                    "nodes": 1,
                    "ntasks_per_node": 1
                },
                "policy": {"auto_retry": False}
            }
            write_yaml_file(cfg_path, cfg)
            print(f"[init] wrote {cfg_path}")

        # 3) run a probe job on compute node
        use_account = account or load_yaml(args.config).get("account")
        use_part = part or load_yaml(args.config).get("default_partition", "gpu")
        use_gres = gres or load_yaml(args.config).get("defaults", {}).get("gres")
        probe = run_probe_job(use_part, use_account, use_gres)
        print("\n[probe] submitted job", probe["job_id"])
        print(f"[probe] stdout: {probe['stdout']}")
        print(f"[probe] stderr: {probe['stderr']}")
        print("----- probe tail (stdout) -----")
        print(probe["out"])

        # 4) quick guidance from probe
        advice = []
        if "nvidia-smi not available" in probe["out"]:
            advice.append("No GPU visible on the chosen partition. Pick a GPU partition or remove GRES.")
        if "PWD_LOGIN_SIDE:" in probe["out"] and "PWD_COMPUTE_SIDE:" in probe["out"]:
            try:
                login_pwd = re.search(r"PWD_LOGIN_SIDE:\s*(.*)", probe["out"]).group(1).strip()
                compute_pwd = re.search(r"PWD_COMPUTE_SIDE:\s*(.*)", probe["out"]).group(1).strip()
                if os.path.realpath(login_pwd) != compute_pwd:
                    advice.append(f"Working dir differs on compute node ({compute_pwd}). Prefer a shared path like $HOME or $SCRATCH.")
            except Exception:
                pass
        if advice:
            print("\n[advice]")
            for a in advice: print("-", a)

        print("\n[init] done ✓")
        sys.exit(0)

    if not args.runner:
        print("usage: slurm python script.py [args..."); sys.exit(1)
    command_line = " ".join(args.runner)

    cfg = load_yaml(args.config)
    defaults = cfg.get("defaults", {})
    env_cfg = cfg.get("env", {})
    log_cfg = cfg.get("logging", {})
    partition = args.partition or cfg.get("default_partition", "gpu")
    account = cfg.get("account")

    # resource guess
    if args.ai:
        r, explain = guess_resources_with_gemini(command_line, os.getcwd(), defaults)
    else:
        r, explain = guess_resources_with_rules(command_line, defaults)

    # explicit overrides
    if args.gpus is not None: r["gpus"] = args.gpus
    if args.mem is not None: r["mem"] = args.mem
    if args.cpus is not None: r["cpus"] = args.cpus
    if args.time is not None: r["time"] = args.time
    if args.partition is not None: partition = args.partition

    # if gres is present, drop gpus
    if r.get("gres") and r.get("gpus"):
        r["gpus"] = None

    # paths
    job_name = args.name or guess_job_name(command_line)
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # temporary job_id unknown yet → use "pending"
    log_dir = log_cfg.get("dir","runs/{date}/{job_name}").format(date=today, job_name=job_name)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stdout = log_cfg.get("stdout","out_%j.log")
    stderr = log_cfg.get("stderr","err_%j.log")

    # render
    ctx = dict(
        job_name=job_name,
        partition=r.get("partition") or partition,
        account=account,
        time=r["time"], nodes=defaults.get("nodes",1),
        ntasks_per_node=defaults.get("ntasks_per_node",1),
        cpus=r["cpus"], mem=r["mem"],
        gpus=r.get("gpus"), gres=r.get("gres"),
        log_dir=str(Path(log_dir).resolve()),
        stdout=stdout, stderr=stderr,
        modules=env_cfg.get("modules", []),
        conda_env=env_cfg.get("conda_env","base"),
        workdir=os.getcwd(),
        command_line=command_line
    )
    script_text = render_template(ROOT/"templates"/"job.sh.j2", ctx)
    script_path = Path(log_dir) / "job.sh"
    script_path.write_text(script_text)
    os.chmod(script_path, 0o755)

    print("\n[plan]")
    print(json.dumps({
        "partition": ctx["partition"], "account": account,
        "gpus": r.get("gpus"), "mem": r["mem"], "cpus": r["cpus"], "time": r["time"],
        "gres": r.get("gres"), "job_name": job_name, "script": str(script_path)
    }, indent=2))

    if args.explain or args.dry_run:
        print("\n[explain]")
        if isinstance(explain, dict):
            for k, v in explain.items():
                print(f"- {k}: {v}")
        else:
            for v in explain:
                print(f"- {v}")

    if args.dry_run:
        print(f"\n[ok] dry-run; script written to {script_path}")
        print(f"\n--- job.sh ---\n{script_text}\n--- end ---")
        return

    # submit
    out = sh(f"sbatch --parsable {script_path}")
    print(f"[submit] {out}")
    job_id = out.split(";")[0].strip()

    final_dir = Path(log_dir)
    state = dict(job_id=job_id, submitted=today, command=command_line, log_dir=log_dir)
    Path(final_dir/"state.json").write_text(json.dumps(state, indent=2))
    print(f"[state] {final_dir}/state.json")

    # monitor: poll squeue; tail logs
    out_log = Path(log_dir)/stdout.replace("%j", job_id)
    err_log = Path(log_dir)/stderr.replace("%j", job_id)
    print(f"[logs]\n  stdout: {out_log}\n  stderr: {err_log}\n")

    last_tail = ""
    while True:
        try:
            status = sh(f"squeue -j {job_id} -h -o '%T %M %R'", check=False)  # State Elapsed Reason
        except Exception:
            status = ""
        if status:
            print(f"[squeue] {job_id}: {status}")
            # stream last 10 lines of stdout
            if out_log.exists():
                tail = sh(f"tail -n 10 {out_log}", check=False)
                if tail != last_tail:
                    print("----- stdout tail -----")
                    print(tail)
                    last_tail = tail
            time.sleep(10)
        else:
            # job exited → get sacct
            acct = sh(f"sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem -P", check=False)
            print("[sacct]\n" + acct)
            # quick triage
            collected = ""
            if err_log.exists():
                collected = (err_log.read_text()[-4000:] if err_log.stat().st_size>0 else "")
            if not collected and out_log.exists():
                collected = (out_log.read_text()[-4000:] if out_log.stat().st_size>0 else "")
            if collected:
                cls = classify_error(collected)
                base_cmd = f"{sys.executable} {Path(__file__).name} " + command_line
                if cls == "OOM":
                    print(f"[triage] Try: --mem 64G  OR  --gpus 2  (e.g., python slurm.py --mem 64G --gpus 2 {command_line})")
                elif cls == "TIME_LIMIT":
                    print(f"[triage] Try: --time 08:00:00  (e.g., python slurm.py --time 08:00:00 {command_line})")
                elif cls == "GRES":
                    print("[triage] GRES issue. Check --gpus vs --gres. Some clusters require only one of them.")
                elif cls == "ACCOUNT/QOS":
                    print("[triage] Account/QOS issue. Verify --account and partition/QOS settings.")
            break

if __name__ == "__main__":
    main()
