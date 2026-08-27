#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
site_packages="${script_dir}/.runtime/geometry/site-packages"

if [[ ! -d "${site_packages}" ]]; then
    echo "Geometry runtime is missing; run ${script_dir}/install_geometry_tools.sh first." >&2
    exit 2
fi

export PYTHONPATH="${site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
exec /usr/bin/python3 "$@"
