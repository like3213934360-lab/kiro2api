#!/bin/bash
# 卸载 macOS 登录自启动
set -e
LABEL="com.kiro.gateway"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
rm -f "$PLIST"
echo "已卸载登录自启动（$LABEL）。"
echo "之后可以用 ./start.sh / ./stop.sh 手动控制。"
