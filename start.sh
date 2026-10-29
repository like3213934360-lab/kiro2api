#!/bin/bash
# 启动 Kiro Gateway（后台运行，日志写入 server.log）
cd "$(dirname "$0")" || exit 1
PID_FILE="gateway.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Kiro Gateway 已经在运行（PID $(cat "$PID_FILE")）：http://127.0.0.1:8000"
  exit 0
fi

if curl -s -m 2 http://127.0.0.1:8000/ >/dev/null 2>&1; then
  echo "端口 8000 上已经有 Kiro Gateway 在运行：http://127.0.0.1:8000"
  exit 0
fi

nohup .venv/bin/python main.py > server.log 2>&1 &
echo $! > "$PID_FILE"
sleep 3

if curl -s -m 3 http://127.0.0.1:8000/ >/dev/null 2>&1; then
  echo "已启动，PID=$(cat "$PID_FILE")"
  echo "地址：http://127.0.0.1:8000   管理后台：http://127.0.0.1:8000/admin"
else
  echo "启动可能失败，请查看日志：$(pwd)/server.log"
  exit 1
fi
