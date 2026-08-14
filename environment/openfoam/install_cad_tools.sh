#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
target_dir="$script_dir/.runtime/cadquery-ocp/site-packages"
package="cadquery-ocp-novtk==7.9.3.1.1"

if PYTHONPATH="$target_dir${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
  'import OCP; assert OCP.__version__ == "7.9.3.1"; from OCP.STEPControl import STEPControl_Reader' \
  >/dev/null 2>&1; then
    printf 'CAD runtime is ready: %s\n' "$target_dir"
    exit 0
fi

mkdir -p "$target_dir"
python3 -m pip install \
  --disable-pip-version-check \
  --upgrade \
  --target "$target_dir" \
  "$package"

PYTHONPATH="$target_dir${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
  'import OCP; assert OCP.__version__ == "7.9.3.1"; from OCP.STEPControl import STEPControl_Reader'
printf 'Installed %s in %s\n' "$package" "$target_dir"
