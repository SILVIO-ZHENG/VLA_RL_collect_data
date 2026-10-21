#!/usr/bin/env bash
set -euo pipefail

ROBOT_SSH="${ROBOT_SSH:-g2}"
ROBOT_IP="${ROBOT_IP:-192.168.110.235}"
SERVER_IP="${SERVER_IP:-10.42.1.200}"
PING_COUNT="${PING_COUNT:-30}"
IPERF_SECONDS="${IPERF_SECONDS:-5}"
IPERF_PORT="${IPERF_PORT:-5202}"
OUTPUT_FILE="${1:-outputs/g2a_pi05_deploy/network/link_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
exec > >(tee "$OUTPUT_FILE") 2>&1

echo "G2A network diagnosis"
date --iso-8601=seconds
echo "robot_ssh=$ROBOT_SSH robot_ip=$ROBOT_IP server_ip=$SERVER_IP"

SERVER_ROUTE="$(ip route get "$ROBOT_IP")"
SERVER_IFACE="$(awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}' <<<"$SERVER_ROUTE")"
ROBOT_ROUTE="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_SSH" "ip route get '$SERVER_IP'")"
ROBOT_IFACE="$(awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}' <<<"$ROBOT_ROUTE")"

echo
echo "Server route: $SERVER_ROUTE"
ip -s link show dev "$SERVER_IFACE"

echo
echo "Robot route: $ROBOT_ROUTE"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_SSH" \
  "ip -s link show dev '$ROBOT_IFACE'; iw dev '$ROBOT_IFACE' link 2>/dev/null || true; iw dev '$ROBOT_IFACE' get power_save 2>/dev/null || true"

if [[ "$ROBOT_IFACE" == wlan* ]]; then
  echo "WARNING: robot traffic is using Wi-Fi interface $ROBOT_IFACE; wired Ethernet is preferred."
fi

echo
echo "Server -> robot ping"
ping -c "$PING_COUNT" -i 0.2 -W 2 "$ROBOT_IP" || true

echo
echo "Robot -> server ping"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_SSH" \
  "ping -c '$PING_COUNT' -i 0.2 -W 2 '$SERVER_IP'" || true

if command -v iperf3 >/dev/null 2>&1 \
  && ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_SSH" 'command -v iperf3 >/dev/null 2>&1'; then
  run_iperf_server() {
    iperf3 -s -1 -p "$IPERF_PORT" >/dev/null &
    IPERF_SERVER_PID=$!
    sleep 0.3
  }

  echo
  echo "iperf3 robot -> server"
  run_iperf_server
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_SSH" \
    "iperf3 -c '$SERVER_IP' -p '$IPERF_PORT' -t '$IPERF_SECONDS' --get-server-output"
  wait "$IPERF_SERVER_PID"

  echo
  echo "iperf3 server -> robot (reverse mode)"
  run_iperf_server
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_SSH" \
    "iperf3 -c '$SERVER_IP' -p '$IPERF_PORT' -t '$IPERF_SECONDS' -R --get-server-output"
  wait "$IPERF_SERVER_PID"
else
  echo
  echo "WARNING: iperf3 is unavailable on one endpoint; skipped bidirectional throughput test."
  echo "Server: $(command -v iperf3 || echo missing)"
  echo "Robot: $(ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT_SSH" 'command -v iperf3 || echo missing')"
fi

echo
echo "diagnosis_log=$OUTPUT_FILE"
