#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_dir="${AUV_OPENFOAM_PACKAGE_DIR:-/tmp}"
runtime_dir="${AUV_OPENFOAM_ROOT:-${script_dir}/.runtime/openfoam2512}"

usage() {
    printf '%s\n' \
        "用法: $0 [--package-dir DIR] [--runtime-dir DIR]" \
        "" \
        "把已下载的 OpenCFD OpenFOAM v2512 Debian 包解压为无需 sudo 的本地运行时。" \
        "目标目录已存在且完整时，本命令只做幂等验证，不会覆盖文件。"
}

while (($#)); do
    case "$1" in
        --package-dir)
            [[ $# -ge 2 ]] || { echo "--package-dir 缺少参数" >&2; exit 2; }
            package_dir="$2"
            shift 2
            ;;
        --runtime-dir)
            [[ $# -ge 2 ]] || { echo "--runtime-dir 缺少参数" >&2; exit 2; }
            runtime_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

project_bashrc="${runtime_dir}/usr/lib/openfoam/openfoam2512/etc/bashrc"
project_pimple="${runtime_dir}/usr/lib/openfoam/openfoam2512/platforms/linux64GccDPInt32Opt/bin/pimpleFoam"
dependency_lib="${runtime_dir}/usr/lib/x86_64-linux-gnu/libscotch-6.1.so"

if [[ -f "${project_bashrc}" && -x "${project_pimple}" && -e "${dependency_lib}" ]]; then
    echo "OpenFOAM v2512 本地运行时已存在：${runtime_dir}"
    exit 0
fi
if [[ -e "${runtime_dir}" ]]; then
    echo "目标目录存在但安装不完整，拒绝覆盖：${runtime_dir}" >&2
    echo "请人工检查后移走该目录，再重新执行安装。" >&2
    exit 1
fi

for command_name in dpkg-deb mktemp; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "缺少安装命令：${command_name}" >&2
        exit 1
    }
done

packages=(
    "openfoam2512-common_2512.0-2_all.deb"
    "openfoam2512-source_2512.0-2_all.deb"
    "openfoam2512-tutorials_2512.0-2_all.deb"
    "openfoam2512_2512.0-2_amd64.deb"
    "libfftw3-double3_3.3.8-2ubuntu8_amd64.deb"
    "libscotch-6.1_6.1.3-1_amd64.deb"
    "libptscotch-6.1_6.1.3-1_amd64.deb"
)
for package_name in "${packages[@]}"; do
    package_path="${package_dir}/${package_name}"
    if [[ ! -f "${package_path}" ]]; then
        echo "缺少离线包：${package_path}" >&2
        exit 1
    fi
done

runtime_parent="$(dirname -- "${runtime_dir}")"
mkdir -p -- "${runtime_parent}"
staging_dir="$(mktemp -d "${runtime_parent}/.openfoam2512.install.XXXXXX")"
cleanup_staging() {
    if [[ -n "${staging_dir:-}" && -d "${staging_dir}" ]]; then
        rm -rf -- "${staging_dir}"
    fi
}
trap cleanup_staging EXIT

for package_name in "${packages[@]}"; do
    echo "解压 ${package_name}"
    dpkg-deb --extract "${package_dir}/${package_name}" "${staging_dir}"
done

staged_bashrc="${staging_dir}/usr/lib/openfoam/openfoam2512/etc/bashrc"
staged_pimple="${staging_dir}/usr/lib/openfoam/openfoam2512/platforms/linux64GccDPInt32Opt/bin/pimpleFoam"
staged_dependency="${staging_dir}/usr/lib/x86_64-linux-gnu/libscotch-6.1.so"
if [[ ! -f "${staged_bashrc}" || ! -x "${staged_pimple}" || ! -e "${staged_dependency}" ]]; then
    echo "解包后缺少 OpenFOAM 核心文件，安装中止。" >&2
    exit 1
fi

mv -- "${staging_dir}" "${runtime_dir}"
staging_dir=""
trap - EXIT

echo "OpenCFD OpenFOAM v2512 已部署到：${runtime_dir}"
echo "加载命令：source ${script_dir}/env.sh"
