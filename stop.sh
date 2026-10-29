#!/bin/bash
# 停止 Kiro Gateway
cd "$(dirname "$0")" || exit 1
PID_FILE="gateway.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "没有 PID 文件，Kiro Gateway 未在运行"
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
