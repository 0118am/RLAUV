#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${script_dir}/env.sh"

if [[ "${FOAM_API:-}" != "2512" || "${WM_PROJECT_VERSION:-}" != "v2512" ]]; then
    echo "版本错误：FOAM_API=${FOAM_API:-unset}, WM_PROJECT_VERSION=${WM_PROJECT_VERSION:-unset}" >&2
    exit 1
fi
if [[ "${WM_PROJECT_DIR}" == /tmp/* ]]; then
    echo "拒绝临时部署：WM_PROJECT_DIR=${WM_PROJECT_DIR}" >&2
    exit 1
fi

python3 "${script_dir}/tools/check_environment.py" --strict --min-api 2512 >/dev/null

for application in blockMesh snappyHexMesh surfaceCheck surfaceTransformPoints pimpleFoam decomposePar; do
    application_path="$(command -v "${application}")"
    missing_libraries="$(ldd "${application_path}" 2>&1 | awk '/not found/')"
    if [[ -n "${missing_libraries}" ]]; then
        echo "${application} 缺少动态库：" >&2
        echo "${missing_libraries}" >&2
        exit 1
    fi
    "${application}" -help >/dev/null
done

installation_report="$(foamInstallationTest 2>&1)"
if [[ "${installation_report}" != *"Base configuration ok"* || \
      "${installation_report}" != *"Critical systems ok"* ]]; then
    echo "foamInstallationTest 未通过：" >&2
    echo "${installation_report}" >&2
    exit 1
fi

printf '%s\n' \
    "OPENFOAM_LOCAL_INSTALL_OK" \
    "version=${WM_PROJECT_VERSION}" \
    "api=${FOAM_API}" \
    "project_dir=${WM_PROJECT_DIR}" \
    "pimpleFoam=$(command -v pimpleFoam)"
