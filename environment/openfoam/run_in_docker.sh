#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
source_stl="${AUV_STL_PATH:-/home/jining_yang/Downloads/auv_visual.stl}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker 不在 PATH 中；可安装 OpenCFD v2512 后直接 source environment/openfoam/env.sh。" >&2
    exit 1
fi
if [[ ! -f "${source_stl}" ]]; then
    echo "找不到 STL: ${source_stl}；请设置 AUV_STL_PATH。" >&2
    exit 1
fi

if (($# == 0)); then
    set -- bash
fi

exec docker run --rm -it \
    --user "$(id -u):$(id -g)" \
    --volume "${repo_dir}:/workspace" \
    --volume "${source_stl}:/input/auv_visual.stl:ro" \
    --workdir /workspace \
    opencfd/openfoam-default:2512 "$@"
