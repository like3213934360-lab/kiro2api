#!/bin/bash
# 无感重启：新进程先接管端口，再让旧进程优雅退出。
#
# 前提是当前进程带 SERVER_REUSE_PORT 启动（start.sh / 本脚本都会设置）。
# 如果旧进程没有这个标志，脚本会自动退回"先停后起"的短空档重启，并提示原因。

set -u
cd "$(dirname "$0")" || exit 1

PID_FILE="gateway.pid"
PORT=$(grep -E '^SERVER_PORT' .env 2>/dev/null | sed -E 's/.*"([0-9]+)".*/\1/')
PORT=${PORT:-8000}
HEALTH="http://127.0.0.1:${PORT}/health"

listeners() { lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null; }

wait_health() {
  local i
  for i in $(seq 1 150); do
    if curl -s -m 2 "$HEALTH" >/dev/null 2>&1; then return 0; fi
    sleep 0.2
  done
  return 1
}

wait_gone() {
  local pid=$1 i
  for i in $(seq 1 450); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

start_new() {
  # Remember where the log ended so the new process's own startup lines can be
  # isolated. lsof cannot be used to confirm the bind: with SO_REUSEPORT the
  # kernel keeps one inode and lsof reports a single PID for both processes.
  LOG_OFFSET=$(wc -c < server.log 2>/dev/null || echo 0)
  SERVER_REUSE_PORT=1 nohup .venv/bin/python main.py >> server.log 2>&1 &
  echo $! > "$PID_FILE"
  NEW_PID=$(cat "$PID_FILE")
}

bound_with_reuse_port() {
  tail -c +$((LOG_OFFSET + 1)) server.log 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -q "Listening socket bound with SO_REUSEPORT"
}

OLD_PID=$(listeners | head -1)

if [ -z "$OLD_PID" ]; then
  echo "没有在跑的实例，直接启动..."
  start_new
  wait_health && echo "已启动，PID=${NEW_PID}" || { echo "启动失败，见 server.log"; exit 1; }
  exit 0
fi

echo "当前实例 PID=${OLD_PID}，启动新实例接管端口..."
start_new

# Wait for the NEW process specifically. Checking /health here would pass
# immediately because the old process is still answering.
BOUND=0
for i in $(seq 1 400); do
  if ! kill -0 "$NEW_PID" 2>/dev/null; then break; fi
  if bound_with_reuse_port; then BOUND=1; break; fi
  sleep 0.1
done

if [ "$BOUND" != "1" ]; then
  echo "新实例无法与旧实例共存（旧实例缺少 SO_REUSEPORT），改用短空档重启"
  kill -TERM "$NEW_PID" 2>/dev/null
  wait_gone "$NEW_PID" || kill -9 "$NEW_PID" 2>/dev/null
  kill -TERM "$OLD_PID" 2>/dev/null
  wait_gone "$OLD_PID" || kill -9 "$OLD_PID" 2>/dev/null
  for i in $(seq 1 100); do [ -z "$(listeners)" ] && break; sleep 0.1; done
  start_new
  if wait_health; then
    echo "已重启，PID=${NEW_PID}（本次有空档，之后的重启将无感）"
    exit 0
  fi
  echo "重启失败，见 server.log"
  exit 1
fi

if ! wait_health; then
  echo "新实例已绑定端口但健康检查未通过，保留旧实例继续服务"
  kill -TERM "$NEW_PID" 2>/dev/null
  exit 1
fi

echo "新实例 PID=${NEW_PID} 已在监听，等待旧实例处理完在途请求..."
# Give the kernel a moment to hand pending connections to the new process before
# the old listening socket goes away.
sleep 1
kill -TERM "$OLD_PID" 2>/dev/null || true
if wait_gone "$OLD_PID"; then
  echo "旧实例 PID=${OLD_PID} 已优雅退出"
else
  echo "旧实例 30s 未退出，强制结束"
  kill -9 "$OLD_PID" 2>/dev/null || true
fi

echo "无感重启完成，PID=${NEW_PID}"
curl -s -m 3 "$HEALTH" >/dev/null 2>&1 && echo "健康检查通过：${HEALTH}"
