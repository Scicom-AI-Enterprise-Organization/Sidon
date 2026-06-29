#!/usr/bin/env bash
# Server-side one-shot: (re)launch the call-centre decoder at batch=6 accum=4 win=6
# (188M, bf16). All the kill/clear/config lives here so the SSH launch command can
# be tiny (just `setsid bash this`) and survive the flaky proxy.
set -u
pkill -9 -f train_decoder_callcentre 2>/dev/null
pkill -9 -f run_decoder 2>/dev/null
sleep 3
# keep last.pt so this RESUMES (don't wipe progress on relaunch)
cd /Sidon
# batch=6/win=6 OOMs (~79 GB, right at the edge); batch=4 accum=6 keeps the SAME
# effective batch (24) and win=6 within memory (~53 GB).
# Scale-up: resume the decoder from last.pt onto the EXPANDED teacher corpus
# (EARS+Expresso + clean_extra) and train to 100k steps. Fresh wandb run name so the
# resumed-from-checkpoint steps (< the old run's high-water mark) aren't dropped by
# wandb's monotonic-step rule.
exec env STEPS=100000 BATCH=4 ACCUM=6 WIN=6 NUM_WORKERS=8 \
    DEC_CHANNELS=3072 WANDB_NAME=decoder-callcentre-3072-realdeg \
    bash runpod/run_decoder_callcentre.sh
