#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
source_surface="${AUV_SURFACE_PATH:-${script_dir}/geometry/validated_locked_rotor_v1/wetted_body_m.obj}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker 不在 PATH 中；可安装 OpenCFD v2512 后直接 source environment/openfoam/env.sh。" >&2
    exit 1
fi
if [[ ! -f "${source_surface}" ]]; then
    echo "找不到 OBJ: ${source_surface}；请设置 AUV_SURFACE_PATH。" >&2
    exit 1
fi

if (($# == 0)); then
    set -- bash
fi

exec docker run --rm -it \
    --user "$(id -u):$(id -g)" \
    --volume "${repo_dir}:/workspace" \
    --volume "${source_surface}:/input/t60_auv_wetted_body_m.obj:ro" \
    --workdir /workspace \
    opencfd/openfoam-default:2512 "$@"
