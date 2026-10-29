#!/bin/bash
# 停止 Kiro Gateway（同时处理 launchd 和手动启动两种方式）
cd "$(dirname "$0")" || exit 1

LABEL="com.kiro.gateway"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"
PID_FILE="gateway.pid"

if [ -f "$PLIST" ] && launchctl print "gui/$UID_NUM/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "gui/$UID_NUM" "$PLIST" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  echo "已停止（launchd）"
  echo "下次开机仍会自动启动；临时启动可运行 ./start.sh。"
  exit 0
fi

if [ ! -f "$PID_FILE" ]; then
  echo "Kiro Gateway 未在运行"
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  sleep 2
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
  echo "已停止（PID $PID）"
else
  echo "进程 $PID 已不存在"
fi

rm -f "$PID_FILE"
