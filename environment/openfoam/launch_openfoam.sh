#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${script_dir}/env.sh"

if [[ "${1:-}" == "--version" || "${1:-}" == "-version" ]]; then
    printf 'OpenCFD OpenFOAM %s (API %s)\n' "${WM_PROJECT_VERSION}" "${FOAM_API}"
    exit 0
fi

if (($#)); then
    "$@"
    exit $?
fi

printf '%s\n' \
    "OpenCFD OpenFOAM ${WM_PROJECT_VERSION} 已加载。" \
    "WM_PROJECT_DIR=${WM_PROJECT_DIR}" \
    "输入 exit 可退出 OpenFOAM shell。"
exec bash --noprofile --rcfile "${script_dir}/env.sh" -i
