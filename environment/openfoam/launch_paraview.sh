#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--case" ]]; then
    [[ $# -ge 2 ]] || { echo "--case 缺少案例目录" >&2; exit 2; }
    case_dir="$2"
    shift 2
    # shellcheck disable=SC1091
    source "${script_dir}/env.sh"
    exec paraFoam -vtk -case "${case_dir}" -- "$@"
fi

exec "${script_dir}/bin/paraview" "$@"
