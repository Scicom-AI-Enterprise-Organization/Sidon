#!/usr/bin/env python3
"""RunPod control CLI for the Sidon dry-run finetune.

Provisions / inspects / tears down a single GPU pod via the RunPod REST API
(https://rest.runpod.io/v1). We boot a stock PyTorch image, inject our SSH
public key so we can drive it over SSH, request a big CONTAINER disk (mounted at
`/`, NOT the network volume at /workspace), and 1x H100.

Reads RUNPOD_API_KEY (and optional overrides) from the environment or the .env
file in the repo root. Only stdlib is used so it runs anywhere.

Commands:
  launch     create the pod, wait until RUNNING + SSH is up, print ssh coords
  status     show the pod's status + SSH endpoint
  ssh        print the ready-to-paste ssh command
  terminate  delete the pod

Pod metadata is cached in runpod/pod.json so the other commands work without
re-passing the id.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POD_JSON = Path(__file__).resolve().parent / "pod.json"
API_BASE = os.environ.get("RUNPOD_API_BASE", "https://rest.runpod.io/v1").rstrip("/")

# Stock RunPod PyTorch image (CUDA 12.8 / torch 2.8 / ubuntu 24.04). Its start.sh
# applies $PUBLIC_KEY to /root/.ssh/authorized_keys and starts sshd, so we get
# SSH for free. The Sidon venv is built fresh with uv on top of this.
DEFAULT_IMAGE = os.environ.get(
    "RUNPOD_IMAGE", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
)
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"


# --------------------------------------------------------------------------- #
# .env loading + HTTP helpers (stdlib only)
# --------------------------------------------------------------------------- #
def load_dotenv() -> None:
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY missing (set it in .env or the environment)")
    return key


def api(method: str, path: str, body: dict | None = None, base: str | None = None) -> dict:
    url = (base or API_BASE) + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {_api_key()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"RunPod API {method} {path} failed: {e.code} {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        sys.exit(f"RunPod API {method} {path} network error: {e}")


# --------------------------------------------------------------------------- #
# SSH endpoint extraction
# --------------------------------------------------------------------------- #
def extract_ssh(pod: dict) -> tuple[str | None, int | None]:
    public_ip = pod.get("publicIp") or pod.get("public_ip")
    pms = pod.get("portMappings")
    if isinstance(pms, dict):
        for k, v in pms.items():
            try:
                if int(k) == 22 and v:
                    return public_ip, int(v)
            except (TypeError, ValueError):
                continue
    if isinstance(pms, list):
        for pm in pms:
            if isinstance(pm, dict) and (pm.get("privatePort") == 22):
                pub = pm.get("publicPort")
                if pub:
                    return pm.get("ip") or public_ip, int(pub)
    runtime = pod.get("runtime") or {}
    for p in runtime.get("ports") or []:
        if isinstance(p, dict) and p.get("privatePort") == 22:
            pub = p.get("publicPort")
            if pub:
                return p.get("ip") or public_ip, int(pub)
    return public_ip, None


def read_pubkey(path: str) -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        sys.exit(f"SSH public key not found: {p}")
    return p.read_text().strip()


def save_pod(meta: dict) -> None:
    POD_JSON.write_text(json.dumps(meta, indent=2))


def load_pod() -> dict:
    if not POD_JSON.exists():
        sys.exit(f"no pod metadata at {POD_JSON} — run `launch` first or pass --pod-id")
    return json.loads(POD_JSON.read_text())


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_launch(a: argparse.Namespace) -> None:
    pubkey = read_pubkey(a.pubkey)
    body = {
        "name": a.name,
        "imageName": a.image,
        "gpuTypeIds": [a.gpu],
        "gpuCount": a.gpu_count,
        "cloudType": a.cloud_type.upper(),
        # Container disk == "/" (local NVMe). volumeInGb=0 => no /workspace
        # network volume, exactly as requested.
        "containerDiskInGb": a.disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pubkey},
    }
    if a.allowed_cuda:
        body["allowedCudaVersions"] = [v.strip() for v in a.allowed_cuda.split(",") if v.strip()]
    print(f"[launch] creating pod {a.name!r}: {a.gpu_count}x {a.gpu}, "
          f"{a.disk_gb}GB container disk, cloud={a.cloud_type}, image={a.image}")
    resp = api("POST", "/pods", body)
    pod_id = resp.get("id")
    if not pod_id:
        sys.exit(f"provision response missing id: {resp}")
    meta = {
        "pod_id": pod_id,
        "name": a.name,
        "gpu": a.gpu,
        "cost_per_hr": resp.get("costPerHr"),
        "ssh_key": os.path.expanduser(a.pubkey).replace(".pub", ""),
    }
    save_pod(meta)
    print(f"[launch] pod_id={pod_id} cost={resp.get('costPerHr')}/hr — waiting for RUNNING + SSH …")
    if a.no_wait:
        return
    wait_for_ssh(pod_id, meta, timeout=a.timeout)


def wait_for_ssh(pod_id: str, meta: dict, timeout: int = 900) -> tuple[str, int]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = api("GET", f"/pods/{pod_id}")
        status = pod.get("desiredStatus") or pod.get("status")
        ip, port = extract_ssh(pod)
        if status == "RUNNING" and ip and port:
            meta.update({"ip": ip, "ssh_port": port,
                         "cost_per_hr": pod.get("costPerHr", meta.get("cost_per_hr"))})
            save_pod(meta)
            key = meta.get("ssh_key", "~/.ssh/id_rsa")
            print(f"[launch] RUNNING  ip={ip} port={port}")
            print(f"[launch] ssh -p {port} -i {key} "
                  f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@{ip}")
            return ip, port
        print(f"  … status={status} ip={ip} port={port}  ({int(deadline - time.time())}s left)")
        time.sleep(10)
    sys.exit("[launch] timed out waiting for RUNNING + SSH endpoint")


def cmd_status(a: argparse.Namespace) -> None:
    pod_id = a.pod_id or load_pod()["pod_id"]
    pod = api("GET", f"/pods/{pod_id}")
    ip, port = extract_ssh(pod)
    print(json.dumps({
        "pod_id": pod_id,
        "name": pod.get("name"),
        "desiredStatus": pod.get("desiredStatus"),
        "ip": ip, "ssh_port": port,
        "costPerHr": pod.get("costPerHr"),
        "gpu": pod.get("machine", {}).get("gpuTypeId"),
    }, indent=2))


def cmd_ssh(a: argparse.Namespace) -> None:
    meta = load_pod()
    pod_id = a.pod_id or meta["pod_id"]
    pod = api("GET", f"/pods/{pod_id}")
    ip, port = extract_ssh(pod)
    if not (ip and port):
        sys.exit("SSH endpoint not ready")
    key = meta.get("ssh_key", "~/.ssh/id_rsa")
    print(f"ssh -p {port} -i {key} -o StrictHostKeyChecking=no "
          f"-o UserKnownHostsFile=/dev/null root@{ip}")


def cmd_terminate(a: argparse.Namespace) -> None:
    pod_id = a.pod_id or load_pod()["pod_id"]
    api("DELETE", f"/pods/{pod_id}")
    print(f"[terminate] pod {pod_id} deleted")
    if POD_JSON.exists() and not a.pod_id:
        POD_JSON.unlink()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("launch", help="create the pod and wait for SSH")
    p.add_argument("--name", default="sidon-dryrun")
    p.add_argument("--gpu", default=DEFAULT_GPU)
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--disk-gb", type=int, default=500)
    p.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--pubkey", default="~/.ssh/id_rsa.pub")
    p.add_argument("--allowed-cuda", default="", help="comma list, e.g. 12.8")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--no-wait", action="store_true")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("status", help="show pod status")
    p.add_argument("--pod-id", default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("ssh", help="print ssh command")
    p.add_argument("--pod-id", default=None)
    p.set_defaults(func=cmd_ssh)

    p = sub.add_parser("terminate", help="delete the pod")
    p.add_argument("--pod-id", default=None)
    p.set_defaults(func=cmd_terminate)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
