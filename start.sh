#!/bin/bash
# 启动 Kiro Gateway（优先使用 macOS LaunchAgent，其次后台运行）
cd "$(dirname "$0")" || exit 1

LABEL="com.kiro.gateway"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"
PID_FILE="gateway.pid"

# Wait for the gateway to answer on /health (startup takes 3-10s: credentials,
# token refresh, model catalog fetch). Polls up to 60s instead of a fixed sleep.
wait_for_server() {
  local i
  for i in $(seq 1 60); do
    if curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if [ -f "$PLIST" ]; then
  if launchctl print "gui/$UID_NUM/$LABEL" >/dev/null 2>&1; then
    launchctl kickstart -k "gui/$UID_NUM/$LABEL"
  else
    launchctl bootstrap "gui/$UID_NUM" "$PLIST"
  fi
  if wait_for_server; then
    echo "已通过 launchd 启动（PID $(launchctl print "gui/$UID_NUM/$LABEL" 2>/dev/null | awk '/pid =/{print $3; exit}')）"
    echo "地址：http://127.0.0.1:8000   管理后台：http://127.0.0.1:8000/admin"
    exit 0
  fi
  echo "launchd 启动可能失败，请查看：$(pwd)/server.log"
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Kiro Gateway 已经在运行（PID $(cat "$PID_FILE")）：http://127.0.0.1:8000"
  exit 0
fi

if curl -s -m 2 http://127.0.0.1:8000/ >/dev/null 2>&1; then
  echo "端口 8000 上已经有 Kiro Gateway 在运行：http://127.0.0.1:8000"
  exit 0
fi

SERVER_REUSE_PORT=1 nohup .venv/bin/python main.py > server.log 2>&1 &
echo $! > "$PID_FILE"
if wait_for_server; then
  echo "已启动，PID=$(cat "$PID_FILE")"
  echo "地址：http://127.0.0.1:8000   管理后台：http://127.0.0.1:8000/admin"
else
  echo "启动可能失败，请查看日志：$(pwd)/server.log"
  tail -n 20 server.log 2>/dev/null
  exit 1
fi
