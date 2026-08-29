#!/usr/bin/env bash
# GPU resume test — validates the full resumed training path on the GPU with NO robot.
#
# What it does (takes ~12-15 minutes):
#   0. kills llama-server (frees the 21.7 GB of GPU; this also drops opencode/Claude Code)
#   1. starts the LEARNER with configs/real/train_config.json (resume: true)
#   2. verifies the resume markers (checkpoint found, weights loaded, NOT from scratch,
#      optimization starts at ~10 Hz with the GPU free)
#   3. lets it train on the restored buffers (online 12000 + offline 10347 frames,
#      replay only — no actor, arms stay off) until it crosses the next save boundary
#      (step 36000), which exercises the policy save + BOTH atomic buffer dumps on the GPU
#   4. verifies the new checkpoint and the re-dumped datasets
#   5. SIGINTs the learner, waits for clean shutdown
#   6. restarts llama-server and waits for it to be healthy
#   7. writes a PASS/FAIL report next to the logs and prints it
#
# Usage (normal terminal, NOT opencode's — opencode dies in step 0 and comes back in step 6):
#   bash /mnt/Storage/projects/hil-serl/gpu_resume_test.sh
#
# Notes:
#   - The step counter advances by ~2000+ replay-only steps; the new checkpoint (0036000+)
#     becomes the next resume point. That is legitimate training on the existing buffers.
#   - All artifacts land in outputs/hilserl_cube_run1/logs/gpu_resume_test_<ts>_*
#   - SAVE_TARGET=38000 MAX_SECONDS=3600 bash gpu_resume_test.sh   (override targets)

set -u

REPO="/mnt/Storage/projects/hil-serl"
OUT="$REPO/outputs/hilserl_cube_run1"
CFG="configs/real/train_config.json"
LOGDIR="$OUT/logs"
TS="$(date +%Y%m%d_%H%M%S)"
STAGE="$LOGDIR/gpu_resume_test_${TS}.log"            # this script's narration
STDOUT_LOG="$LOGDIR/gpu_resume_test_${TS}_learner_stdout.log"
RESULT="$LOGDIR/gpu_resume_test_${TS}_RESULT.txt"

SAVE_TARGET="${SAVE_TARGET:-36000}"
MAX_SECONDS="${MAX_SECONDS:-3600}"     # overall watchdog: learner start -> checkpoint written
STARTUP_TIMEOUT=900                    # resume-load + first optimization step
CHECKPOINT_TIMEOUT=1500                # from first Hz line to checkpoint dir
SHUTDOWN_TIMEOUT=240                   # SIGINT to "Cleanup complete"
LLAMA_TIMEOUT=360                      # llama-server restart

mkdir -p "$LOGDIR"
: > "$STAGE"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STAGE"; }

RESULT_STATUS="FAIL"
write_result() {
    # $1 = status, rest = evidence lines
    local status="$1"; shift
    {
        echo "GPU RESUME TEST — $(date)"
        echo "STATUS: $status"
        echo "run: outputs/hilserl_cube_run1 (resumed from checkpoint ${LAST_STEP:-none}, wandb opp47yi6)"
        echo
        for line in "$@"; do echo "$line"; done
        echo
        echo "narration log : $STAGE"
        echo "learner stdout: $STDOUT_LOG"
        echo "learner log   : $LOGDIR/learner_cube_run1.log"
    } > "$RESULT"
    cat "$RESULT" >> "$STAGE" 2>/dev/null || true
}

finish() {
    local status="${1:-FAIL}"
    write_result "$status" "${EVIDENCE[@]}"
    log "RESULT: $status  ->  $RESULT"
    echo
    cat "$RESULT"
    # Always bring the LLM back (it is a systemd user service: llama-server.service)
    if ! pgrep -f "llama-server" > /dev/null 2>&1; then
        log "Starting llama-server service..."
        if command -v systemctl > /dev/null 2>&1 && systemctl --user show llama-server > /dev/null 2>&1; then
            systemctl --user reset-failed llama-server 2>/dev/null || true
            systemctl --user start llama-server 2>/dev/null || true
        fi
        # Fallback: no unit (or start failed) -> direct launch
        if ! pgrep -f "llama-server" > /dev/null 2>&1; then
            nohup ~/local-llm/run-llama-server.sh > ~/local-llm/logs/llama-server.log 2>&1 &
        fi
        local i
        for i in $(seq 1 $((LLAMA_TIMEOUT / 5))); do
            if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
                log "llama-server is healthy again."
                break
            fi
            sleep 5
        done
        if ! curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
            log "WARNING: llama-server not healthy after ${LLAMA_TIMEOUT}s — check 'journalctl --user -u llama-server' and ~/local-llm/logs/llama-server.log"
        fi
    else
        log "llama-server already running, leaving it alone."
    fi
}

EVIDENCE=()
add_evidence() { EVIDENCE+=("$1"); }

# Learner's own log grows by appending; only look at lines written after we start it.
LEARNER_LOG="$LOGDIR/learner_cube_run1.log"
LOG_OFFSET=0
[ -f "$LEARNER_LOG" ] && LOG_OFFSET=$(stat -c %s "$LEARNER_LOG")

new_learner_log_lines() {
    local cur=0
    [ -f "$LEARNER_LOG" ] && cur=$(stat -c %s "$LEARNER_LOG")
    if [ "$cur" -gt "$LOG_OFFSET" ]; then
        tail -c +"$((LOG_OFFSET + 1))" "$LEARNER_LOG"
    elif [ "$cur" -lt "$LOG_OFFSET" ]; then
        # file was recreated/truncated
        cat "$LEARNER_LOG" 2>/dev/null
    fi
}

# grep both the learner log (new part) and the captured stdout for a pattern
found() {
    local pat="$1"
    grep -qE "$pat" "$STDOUT_LOG" 2>/dev/null && return 0
    new_learner_log_lines | grep -qE "$pat" 2>/dev/null && return 0
    return 1
}

# Stop the learner cleanly: SIGINT, wait for "Cleanup complete" (or exit), then
# escalate to SIGTERM/SIGKILL if it refuses. No-op if it never started or died.
stop_learner() {
    [ -n "${LPID:-}" ] || return 0
    kill -0 "$LPID" 2>/dev/null || return 0
    log "Sending SIGINT to learner..."
    kill -INT "$LPID" 2>/dev/null
    local deadline=$(( $(date +%s) + SHUTDOWN_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        kill -0 "$LPID" 2>/dev/null || break
        found "Cleanup complete" && break
        sleep 5
    done
    if kill -0 "$LPID" 2>/dev/null; then
        log "learner did not exit in ${SHUTDOWN_TIMEOUT}s, sending SIGTERM"
        kill -TERM "$LPID" 2>/dev/null
        sleep 15
        kill -0 "$LPID" 2>/dev/null && kill -9 "$LPID" 2>/dev/null
    fi
    wait "$LPID" 2>/dev/null || true
    log "learner stopped (cleanup complete: $(found "Cleanup complete" && echo yes || echo no))"
}

trap 'stop_learner; finish "FAIL (script aborted)"' INT TERM

# ---------- 0. Preflight ----------
log "=== GPU resume test starting (ts $TS) ==="
command -v nvidia-smi > /dev/null || { log "no nvidia-smi on PATH"; finish "FAIL (no nvidia-smi)"; exit 1; }
[ -e "$OUT/checkpoints/last" ] || { log "no $OUT/checkpoints/last"; finish "FAIL (no checkpoint)"; exit 1; }
LAST_STEP="$(basename "$(readlink -f "$OUT/checkpoints/last" 2>/dev/null)")"
log "current checkpoint: $LAST_STEP"
pgrep -f "lerobot.rl.learner" > /dev/null 2>&1 && { log "a learner is already running"; finish "FAIL (learner already running)"; exit 1; }

# ---------- 1. Stop the LLM (systemd user service, Restart=on-failure) ----------
# A plain pkill makes the service fight us: systemd restarts it, and it grabs
# the GPU back. Stop the unit so it stays dead (and clear any failed state so
# finish() can start it again).
log "Stopping llama-server service..."
if command -v systemctl > /dev/null 2>&1 && systemctl --user show llama-server > /dev/null 2>&1; then
    systemctl --user stop llama-server 2>/dev/null || true
    systemctl --user reset-failed llama-server 2>/dev/null || true
else
    pkill -f "llama-server" 2>/dev/null || true
fi
i=0
while [ $i -lt 18 ]; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]; then
        break
    fi
    sleep 5; i=$((i + 1))
done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]; then
    log "GPU still busy after 90s, forcing kill of compute pids..."
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 10
fi
add_evidence "step 0: llama-server killed, GPU free"
log "GPU is free."

# ---------- 2. Environment ----------
set -a; . "$REPO/.env"; set +a
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
cd "$REPO"

# ---------- 3. Start the learner ----------
log "Starting learner (resume from $LAST_STEP)..."
python -m lerobot.rl.learner --config_path "$CFG" > "$STDOUT_LOG" 2>&1 &
LPID=$!
log "learner PID $LPID"
RUN_DEADLINE=$(( $(date +%s) + MAX_SECONDS ))   # overall watchdog for learner start -> checkpoint
DEAD=0
is_dead() { kill -0 "$LPID" 2>/dev/null || { DEAD=1; return 1; }; return 0; }

# ---------- 4. Wait for resume + first optimization ----------
log "Waiting for resume markers (up to ${STARTUP_TIMEOUT}s)..."
START_EPOCH=$(date +%s)
DEADLINE=$((START_EPOCH + STARTUP_TIMEOUT))
OK_RESUME=0 OK_WEIGHTS=0 OK_GRPC=0 OK_HZ=0 BAD_SCRATCH=0
while [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$(date +%s)" -lt "$RUN_DEADLINE" ]; do
    is_dead || { log "learner process died during startup"; break; }
    found "Valid checkpoint found: resume=True detected" && OK_RESUME=1
    found "Loading weights from local directory" && OK_WEIGHTS=1
    found "gRPC server started" && OK_GRPC=1
    found "Optimization frequency loop \[Hz\]" && OK_HZ=1
    found "instantiating a policy from scratch" && BAD_SCRATCH=1
    [ "$OK_RESUME" = 1 ] && [ "$OK_WEIGHTS" = 1 ] && [ "$OK_GRPC" = 1 ] && [ "$OK_HZ" = 1 ] && break
    sleep 5
done
if [ "$BAD_SCRATCH" = 1 ]; then
    log "CRITICAL: policy was instantiated from scratch — resume is broken"
    stop_learner
    add_evidence "resume: policy instantiated FROM SCRATCH (resume broken)"
    finish "FAIL (resume broken: from scratch)"
    exit 1
fi
add_evidence "resume markers: checkpoint_found=$OK_RESUME weights_loaded=$OK_WEIGHTS grpc=$OK_GRPC first_optimization=$OK_HZ from_scratch=$BAD_SCRATCH"
log "markers: checkpoint=$OK_RESUME weights=$OK_WEIGHTS grpc=$OK_GRPC optimization=$OK_HZ scratch=$BAD_SCRATCH"
if [ "$DEAD" = 1 ] || [ "$OK_HZ" = 0 ]; then
    log "startup did not reach the optimization loop"
    tail -n 30 "$STDOUT_LOG" >> "$STAGE"; new_learner_log_lines | tail -n 30 >> "$STAGE"
    stop_learner
    finish "FAIL (startup)"
    exit 1
fi

# ---------- 5. Wait for the checkpoint at/after SAVE_TARGET ----------
# Match the existing checkpoint dir width (7 digits, e.g. 0034000) so the
# computed path lines up with what the learner actually writes.
CKPT_DIR="$OUT/checkpoints/$(printf "%0${#LAST_STEP}d" "$SAVE_TARGET")"
log "Waiting for checkpoint $CKPT_DIR (up to ${CHECKPOINT_TIMEOUT}s)..."
DEADLINE=$(( $(date +%s) + CHECKPOINT_TIMEOUT ))
OK_CKPT=0
while [ "$(date +%s)" -lt "$DEADLINE" ] && [ "$(date +%s)" -lt "$RUN_DEADLINE" ]; do
    is_dead || { log "learner process died while waiting for checkpoint"; break; }
    if [ -d "$CKPT_DIR" ] && found "Checkpoint policy after step $SAVE_TARGET"; then
        OK_CKPT=1
        break
    fi
    sleep 10
done
add_evidence "checkpoint $SAVE_TARGET: $([ "$OK_CKPT" = 1 ] && echo written || echo MISSING)"
log "checkpoint status: $OK_CKPT"
if [ "$DEAD" = 1 ] || [ "$OK_CKPT" = 0 ]; then
    tail -n 30 "$STDOUT_LOG" >> "$STAGE"; new_learner_log_lines | tail -n 30 >> "$STAGE"
    stop_learner
    finish "FAIL (checkpoint)"
    exit 1
fi

# The "Checkpoint policy after step N" line is logged BEFORE the buffer dumps run;
# the dumps write to .partial and rename into place when done. Wait until both
# dataset dirs hold a parquet newer than the checkpoint start (i.e. renamed).
DUMP_EPOCH=$(date +%s)
log "Waiting for both buffer dumps to land (up to 900s)..."
DEADLINE=$(( $(date +%s) + 900 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    found "Replay-buffer dataset dump failed" && { log "dump failed (see log)"; break; }
    is_dead || { log "learner died during buffer dumps"; break; }
    find "$OUT/dataset/data" -name "*.parquet" -newermt "@$DUMP_EPOCH" 2>/dev/null | grep -q . \
      && find "$OUT/dataset_offline/data" -name "*.parquet" -newermt "@$DUMP_EPOCH" 2>/dev/null | grep -q . \
      && { log "both buffer dumps landed."; break; }
    sleep 10
done

# Run a bit past the checkpoint to confirm the learner survives its own dump,
# and to gather more Hz samples.
log "Learner survived its dump; running 60s more for Hz stats..."
sleep 60
if ! is_dead; then
    log "learner died right after the checkpoint dump"
    tail -n 30 "$STDOUT_LOG" >> "$STAGE"; new_learner_log_lines | tail -n 30 >> "$STAGE"
    finish "FAIL (crash after checkpoint)"
    exit 1
fi

# Hz stats from the new part of the log
HZ_STATS=$(new_learner_log_lines | grep -oE "Optimization frequency loop \[Hz\]: [0-9.]+" | sed 's/.*: //' | sort -n | awk '{a[NR]=$1; t+=$1} END {if (NR) printf "n=%d min=%.1f median=%.1f max=%.1f mean=%.1f", NR, a[1], a[int((NR+1)/2)], a[NR], t/NR}')
add_evidence "optimization Hz: $HZ_STATS"
log "Hz stats: $HZ_STATS"

# ---------- 6. Verify artifacts ----------
DUMP_FAIL=0
new_learner_log_lines | grep -q "Replay-buffer dataset dump failed" && DUMP_FAIL=1
add_evidence "buffer dumps in checkpoint: failed=$DUMP_FAIL"

VERIFY_PY="$LOGDIR/gpu_resume_test_${TS}_verify.py"
cat > "$VERIFY_PY" <<'PYEOF'
import json, sys
import pyarrow.dataset as ds
import pyarrow.compute as pc

OUT = "/mnt/Storage/projects/hil-serl/outputs/hilserl_cube_run1"
from PIL import Image
import io

def check_dataset(name, path, expect_rows):
    d = ds.dataset(path + "/data").to_table()
    rows = d.num_rows
    eps = len(d.column("episode_index").unique().to_pylist())
    sel = pc.field("index").isin([0, rows // 2, rows - 1])
    sub = d.filter(sel)
    ok_img = 0
    for key in ("observation.images.scene", "observation.images.wrist"):
        for v in sub.column(key).to_pylist():
            b = v["bytes"] if isinstance(v, dict) else v
            img = Image.open(io.BytesIO(b)).convert("RGB")
            assert img.size == (128, 128), img.size
            ok_img += 1
    print(f"{name}: rows={rows} episodes={eps} frames_decoded={ok_img} "
          f"({'OK' if rows == expect_rows else 'ROWS MISMATCH expect ' + str(expect_rows)})")
    return rows == expect_rows

ok = True
ok &= check_dataset("dataset (online)", OUT + "/dataset", 12000)
ok &= check_dataset("dataset_offline (demos)", OUT + "/dataset_offline", 10347)
info = json.load(open(OUT + "/checkpoints/" + sys.argv[1] + "/training_state/training_step.json"))
print("training_step.json:", info)
sys.exit(0 if ok else 1)
PYEOF
CKPT_NAME="$(basename "$(readlink -f "$CKPT_DIR")")"
if python "$VERIFY_PY" "$CKPT_NAME" >> "$STAGE" 2>&1; then
    add_evidence "datasets after dump: verified (12000 online / 10347 offline, frames decode to 128x128 RGB)"
else
    add_evidence "datasets after dump: VERIFICATION FAILED (see narration log)"
fi

LEFTOVERS="$(ls "$OUT" | grep -E '\.(partial|old)$' || true)"
add_evidence "leftover .partial/.old dirs: ${LEFTOVERS:-none}"

NEW_LAST="$(basename "$(readlink -f "$OUT/checkpoints/last" 2>/dev/null)")"
add_evidence "checkpoints/last now: $NEW_LAST (was $LAST_STEP)"

# ---------- 7. Shut the learner down ----------
stop_learner

# ---------- 8. Result ----------
STATUS="PASS"
[ "$BAD_SCRATCH" = 1 ] && STATUS="FAIL"
[ "$DUMP_FAIL" = 1 ] && STATUS="FAIL"
[ -n "$LEFTOVERS" ] && STATUS="FAIL"
[ "$NEW_LAST" != "$CKPT_NAME" ] && STATUS="FAIL"
HZ_MEAN="$(echo "$HZ_STATS" | grep -oE 'mean=[0-9.]+' | cut -d= -f2)"
if [ -n "$HZ_MEAN" ]; then
    awk -v h="$HZ_MEAN" 'BEGIN { exit (h >= 7) ? 0 : 1 }' || { STATUS="FAIL"; add_evidence "Hz too low: mean $HZ_MEAN (<7)"; }
else
    STATUS="FAIL"; add_evidence "no Hz stats collected"
fi

finish "$STATUS"
exit 0
