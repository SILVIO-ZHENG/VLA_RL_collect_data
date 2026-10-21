#!/usr/bin/env bash
set -euo pipefail

DURATION_S="${DURATION_S:-30}"
OUTPUT_DIR="${1:-outputs/g2a_pi05_deploy/system}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_DIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Collecting server load for ${DURATION_S}s under $RUN_DIR"

date --iso-8601=seconds >"$RUN_DIR/metadata.txt"
uname -a >>"$RUN_DIR/metadata.txt"
if command -v uptime >/dev/null 2>&1; then
  uptime >>"$RUN_DIR/metadata.txt"
else
  echo -n "uptime_seconds=" >>"$RUN_DIR/metadata.txt"
  cut -d' ' -f1 /proc/uptime >>"$RUN_DIR/metadata.txt"
fi
ip route >>"$RUN_DIR/metadata.txt"
ip -s link >>"$RUN_DIR/network_links.txt"
ps -eo pid,ppid,psr,stat,pcpu,pmem,etimes,comm,args --sort=-pcpu >"$RUN_DIR/processes_start.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"$RUN_DIR/nvidia_smi_start.txt"
  nvidia-smi dmon -s pucvmet -d 1 -c "$DURATION_S" >"$RUN_DIR/nvidia_dmon.txt" &
  GPU_MONITOR_PID=$!
else
  GPU_MONITOR_PID=""
fi

vmstat 1 "$DURATION_S" >"$RUN_DIR/vmstat.txt" &
VMSTAT_PID=$!

if command -v pidstat >/dev/null 2>&1; then
  pidstat -dur -p ALL 1 "$DURATION_S" >"$RUN_DIR/pidstat.txt" &
  PIDSTAT_PID=$!
else
  PIDSTAT_PID=""
fi

wait "$VMSTAT_PID"
if [[ -n "$GPU_MONITOR_PID" ]]; then
  wait "$GPU_MONITOR_PID"
fi
if [[ -n "$PIDSTAT_PID" ]]; then
  wait "$PIDSTAT_PID"
fi

ps -eo pid,ppid,psr,stat,pcpu,pmem,etimes,comm,args --sort=-pcpu >"$RUN_DIR/processes_end.txt"
ip -s link >"$RUN_DIR/network_links_end.txt"
echo "$RUN_DIR"
