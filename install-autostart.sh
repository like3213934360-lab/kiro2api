#!/bin/bash
# 安装/更新 macOS 登录自启动（LaunchAgent）
set -e
cd "$(dirname "$0")" || exit 1
PROJECT_DIR="$(pwd)"
LABEL="com.kiro.gateway"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"

mkdir -p "$HOME/Library/LaunchAgents"

# Stop the current manual instance (if any) so the port is free.
"$PROJECT_DIR/stop.sh" >/dev/null 2>&1 || true

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PROJECT_DIR/.venv/bin/python</string>
        <string>$PROJECT_DIR/main.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/server.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/server.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$UID_NUM" "$PLIST" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST"

sleep 3
if curl -s -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "已安装并启动：http://127.0.0.1:8000"
  echo "管理后台：http://127.0.0.1:8000/admin"
  echo "以后开机登录会自动启动，无需手动操作。"
else
  echo "已安装，但服务还没起来，请查看：$(pwd)/server.log"
  exit 1
fi
