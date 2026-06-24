#!/usr/bin/env bash
# Local driver: rsync this repo to the pod (from runpod/pod.json), bootstrap it,
# and launch the dry-run in the background (survives SSH disconnect).
#
#   ./runpod/sync_and_launch.sh            # sync + bootstrap + launch
#   ./runpod/sync_and_launch.sh sync       # just rsync the code
#   ./runpod/sync_and_launch.sh bootstrap  # just run bootstrap.sh
#   ./runpod/sync_and_launch.sh launch     # just (re)launch the dry run
#   ./runpod/sync_and_launch.sh tail       # tail the run log
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_LOCAL="$(dirname "$HERE")"
POD_JSON="${POD_JSON:-$HERE/pod.json}"
[ -f "$POD_JSON" ] || { echo "no $POD_JSON — run launch_pod.py launch first"; exit 1; }

read IP PORT KEY < <(python3 - "$POD_JSON" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["ip"], m["ssh_port"], m.get("ssh_key", "~/.ssh/id_rsa"))
PY
)
KEY="${KEY/#\~/$HOME}"
SSH="ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@$IP"

do_sync() {
    echo "[sync] rsync -> root@$IP:/Sidon"
    # NOTE: top-level dir excludes are ANCHORED with a leading "/", otherwise an
    # unanchored "data" would also drop config/data (the data-module configs).
    rsync -az --delete \
        -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
        --exclude '/.venv' --exclude '/.git' --exclude '/runs' --exclude '/wandb' \
        --exclude '__pycache__' --exclude '/.hf_cache' --exclude '/data' \
        --exclude '/ckpt_v0.1' --exclude '/outputs' --exclude 'runpod/pod*.json' \
        --exclude '*.ckpt' --exclude '*.pt' --exclude '*.bin' --exclude '*.zip' \
        --exclude '/infer.ipynb' \
        "$REPO_LOCAL/" "root@$IP:/Sidon/"
}

do_bootstrap() { $SSH "bash /Sidon/runpod/bootstrap.sh 2>&1 | tee /Sidon/bootstrap.log"; }

do_launch() {
    echo "[launch] starting dry run in background on pod"
    $SSH "cd /Sidon && nohup bash runpod/run_sidon_dryrun.sh > /Sidon/train.log 2>&1 & echo started pid \$!"
}

do_tail() { $SSH "tail -n 120 -f /Sidon/train.log"; }

case "${1:-all}" in
    sync)      do_sync ;;
    bootstrap) do_bootstrap ;;
    launch)    do_launch ;;
    tail)      do_tail ;;
    all)       do_sync; do_bootstrap; do_launch ;;
    *) echo "usage: $0 [sync|bootstrap|launch|tail|all]"; exit 1 ;;
esac
