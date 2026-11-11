#!/usr/bin/env bash
# Resolve current /dev/dri/renderD* names from stable PCI by-path symlinks.
#
# After a reboot, render node numbers can change (on z8g4 the second B70
# moved from renderD130 to renderD131). PCI addresses do not. Recheck before
# `docker compose up`, or run this script with --write to update .env.
# In-container resolution cannot fix a wrong devices: bind: Docker passes
# render nodes at container create time, before entrypoint runs.
#
# Usage:
#   ./scripts/resolve-dri-from-bypath.sh
#   ./scripts/resolve-dri-from-bypath.sh --env .env
#   ./scripts/resolve-dri-from-bypath.sh --write .env
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: resolve-dri-from-bypath.sh [--env FILE] [--write [FILE]]

  --env FILE    Read DRI_BYPATH_0 / DRI_BYPATH_1 from FILE
                (default: .env if present, else examples/z8g4.env)
  --write FILE  Update DRI_RENDER_* and DRI_RENDER_*_NAME in FILE
                (FILE defaults to the --env file)

PCI paths can also be supplied in the environment as DRI_BYPATH_0 / DRI_BYPATH_1.
USAGE
}

ENV_FILE=""
WRITE=0
WRITE_FILE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --write)
      WRITE=1
      shift
      if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
        WRITE_FILE="$1"
        shift
      fi
      ;;
    --env)
      shift
      ENV_FILE="${1:?--env requires a file}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "$ENV_FILE" ]; then
  if [ -f .env ]; then
    ENV_FILE=.env
  elif [ -f examples/z8g4.env ]; then
    ENV_FILE=examples/z8g4.env
  fi
fi

env_get() {
  local key="$1" file="$2"
  [ -n "$file" ] && [ -f "$file" ] || return 0
  # Last non-comment assignment. Values in this repo are unquoted paths.
  grep -E "^${key}=" "$file" 2>/dev/null | tail -n 1 | sed "s/^${key}=//" || true
}

if [ -z "${DRI_BYPATH_0:-}" ] && [ -n "$ENV_FILE" ]; then
  DRI_BYPATH_0="$(env_get DRI_BYPATH_0 "$ENV_FILE")"
fi
if [ -z "${DRI_BYPATH_1:-}" ] && [ -n "$ENV_FILE" ]; then
  DRI_BYPATH_1="$(env_get DRI_BYPATH_1 "$ENV_FILE")"
fi

if [ -z "${DRI_BYPATH_0:-}" ]; then
  echo "DRI_BYPATH_0 is not set (pass it in the environment or use --env FILE)" >&2
  exit 1
fi

resolve_one() {
  local bypath="$1"
  if [ ! -e "$bypath" ]; then
    echo "missing by-path: $bypath" >&2
    echo "Check ls -l /dev/dri/by-path and the PCI address in DRI_BYPATH_*." >&2
    return 1
  fi
  readlink -f "$bypath"
}

update_key() {
  local file="$1" key="$2" value="$3"
  if grep -qE "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$file"
  fi
}

echo "=== resolved render nodes (from PCI by-path) ==="
if [ -n "$ENV_FILE" ]; then
  echo "env file: $ENV_FILE"
fi

RENDER_0="$(resolve_one "$DRI_BYPATH_0")"
NAME_0="$(basename "$RENDER_0")"
echo "DRI_BYPATH_0=$DRI_BYPATH_0"
echo "DRI_RENDER_0=$RENDER_0"
echo "DRI_RENDER_0_NAME=$NAME_0"

RENDER_1=""
NAME_1=""
if [ -n "${DRI_BYPATH_1:-}" ]; then
  RENDER_1="$(resolve_one "$DRI_BYPATH_1")"
  NAME_1="$(basename "$RENDER_1")"
  echo "DRI_BYPATH_1=$DRI_BYPATH_1"
  echo "DRI_RENDER_1=$RENDER_1"
  echo "DRI_RENDER_1_NAME=$NAME_1"
fi

if [ "$WRITE" = 1 ]; then
  TARGET="${WRITE_FILE:-$ENV_FILE}"
  if [ -z "$TARGET" ]; then
    echo "--write requires a file (pass --write .env or --env FILE)" >&2
    exit 1
  fi
  if [ ! -f "$TARGET" ]; then
    echo "refusing to write missing file: $TARGET" >&2
    exit 1
  fi
  update_key "$TARGET" DRI_RENDER_0 "$RENDER_0"
  update_key "$TARGET" DRI_RENDER_0_NAME "$NAME_0"
  if [ -n "$RENDER_1" ]; then
    update_key "$TARGET" DRI_RENDER_1 "$RENDER_1"
    update_key "$TARGET" DRI_RENDER_1_NAME "$NAME_1"
  fi
  echo
  echo "updated $TARGET"
fi
