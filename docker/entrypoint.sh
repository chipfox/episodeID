#!/bin/sh
# Run as PUID:PGID (Unraid default 99:100) so renamed/split files keep
# ownership your other containers expect.
set -e
PUID=${PUID:-99}
PGID=${PGID:-100}
# Docker's group_add is cleared by setpriv --clear-groups. Keep the render group
# so the app user can open /dev/dri for VAAPI/OpenVINO without running as root.
RENDER_GID=${RENDER_GID:-}
if [ -z "$RENDER_GID" ] && [ -e /dev/dri/renderD128 ]; then
  RENDER_GID=$(stat -c '%g' /dev/dri/renderD128 2>/dev/null || true)
fi
umask "${UMASK:-002}"
mkdir -p "$CONFIG_DIR" "$CACHE_DIR"

if [ "$(id -u)" = "0" ]; then
  chown "$PUID:$PGID" "$CONFIG_DIR" "$CACHE_DIR" 2>/dev/null || true
  chown -R "$PUID:$PGID" "$CONFIG_DIR" "$CACHE_DIR" 2>/dev/null || true
  export HOME="$CONFIG_DIR"
  if [ -n "$RENDER_GID" ]; then
    exec setpriv --reuid="$PUID" --regid="$PGID" --groups="$RENDER_GID" \
      python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8686}" --proxy-headers
  fi
  exec setpriv --reuid="$PUID" --regid="$PGID" --clear-groups \
    python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8686}" --proxy-headers
fi
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8686}" --proxy-headers
