#!/usr/bin/env bash
# Install the HIL-SERL udev rules and show the resulting stable device names.
set -euo pipefail
cd "$(dirname "$0")"
cp 99-so101-hilserl.rules /etc/udev/rules.d/
udevadm control --reload-rules
udevadm trigger
sleep 1
echo "--- stable names now present: ---"
ls -l /dev/so101_* /dev/cam_* 2>&1
