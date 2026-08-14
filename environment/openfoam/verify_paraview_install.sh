#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paraview_root="${AUV_PARAVIEW_ROOT:-${script_dir}/.runtime/paraview-6.0.1}"
paraview_real="${paraview_root}/bin/paraview-real"

version_output="$("${script_dir}/bin/paraview" --version 2>&1)"
if [[ "${version_output}" != *"6.0.1"* ]]; then
    echo "ParaView 版本验证失败：${version_output}" >&2
    exit 1
fi
if [[ ! -x "${paraview_real}" ]]; then
    echo "找不到 paraview-real：${paraview_real}" >&2
    exit 1
fi

missing_libraries="$(ldd "${paraview_real}" 2>&1 | awk '/not found/')"
if [[ -n "${missing_libraries}" ]]; then
    echo "ParaView 缺少动态库：" >&2
    echo "${missing_libraries}" >&2
    exit 1
fi

printf '%s\n' \
    "PARAVIEW_LOCAL_INSTALL_OK" \
    "version=6.0.1" \
    "root=${paraview_root}" \
    "launcher=${script_dir}/launch_paraview.sh"
