#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive_name="ParaView-6.0.1-MPI-Linux-Python3.12-x86_64.tar.gz"
archive_path="${AUV_PARAVIEW_ARCHIVE:-/tmp/${archive_name}}"
xcb_package_name="libxcb-cursor0_0.1.1-4ubuntu1_amd64.deb"
xcb_package_path="${AUV_PARAVIEW_XCB_PACKAGE:-/tmp/${xcb_package_name}}"
runtime_dir="${AUV_PARAVIEW_ROOT:-${script_dir}/.runtime/paraview-6.0.1}"
archive_root="ParaView-6.0.1-MPI-Linux-Python3.12-x86_64"
expected_size=826711252
expected_md5="de2be9dcb3bced49f4959bd932aabc0a"
expected_sha256="62995dc07907d0fe5049244b5fa82b03e9f0fe255159fa690d586367dfcb1959"
expected_xcb_sha256="c9b5d1ad4af57397b1bd77e0a92750e34419def134c0282a0836ae9efc07cf64"

usage() {
    printf '%s\n' \
        "用法: $0 [--archive FILE] [--xcb-package FILE] [--runtime-dir DIR]" \
        "" \
        "将 Kitware 官方 ParaView 6.0.1 Linux 二进制包安装到项目内，无需 sudo。"
}

while (($#)); do
    case "$1" in
        --archive)
            [[ $# -ge 2 ]] || { echo "--archive 缺少参数" >&2; exit 2; }
            archive_path="$2"
            shift 2
            ;;
        --xcb-package)
            [[ $# -ge 2 ]] || { echo "--xcb-package 缺少参数" >&2; exit 2; }
            xcb_package_path="$2"
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

paraview_executable="${runtime_dir}/bin/paraview"
pvpython_executable="${runtime_dir}/bin/pvpython"
xcb_library="${runtime_dir}/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0"
if [[ -x "${paraview_executable}" && -x "${pvpython_executable}" && -e "${xcb_library}" ]]; then
    echo "ParaView 6.0.1 已安装：${runtime_dir}"
    exit 0
fi
if [[ -e "${runtime_dir}" && ( ! -x "${paraview_executable}" || ! -x "${pvpython_executable}" ) ]]; then
    echo "目标目录存在但安装不完整，拒绝覆盖：${runtime_dir}" >&2
    exit 1
fi
if [[ ! -e "${runtime_dir}" && ! -f "${archive_path}" ]]; then
    echo "找不到 ParaView 官方安装包：${archive_path}" >&2
    echo "下载地址：https://www.paraview.org/files/v6.0/${archive_name}" >&2
    exit 1
fi

for command_name in dpkg-deb md5sum mktemp sha256sum stat tar; do
    command -v "${command_name}" >/dev/null 2>&1 || {
        echo "缺少安装命令：${command_name}" >&2
        exit 1
    }
done

if [[ ! -f "${xcb_package_path}" ]]; then
    echo "找不到 Qt XCB 运行库包：${xcb_package_path}" >&2
    echo "可执行：cd /tmp && apt-get download libxcb-cursor0=0.1.1-4ubuntu1" >&2
    exit 1
fi
xcb_sha256="$(sha256sum -- "${xcb_package_path}")"
xcb_sha256="${xcb_sha256%% *}"
if [[ "${xcb_sha256}" != "${expected_xcb_sha256}" ]]; then
    echo "libxcb-cursor0 的 SHA-256 不匹配。" >&2
    exit 1
fi

if [[ -e "${runtime_dir}" ]]; then
    echo "正在为现有 ParaView 安装 libxcb-cursor0..."
    dpkg-deb --extract "${xcb_package_path}" "${runtime_dir}"
    [[ -e "${xcb_library}" ]] || { echo "libxcb-cursor0 解包失败" >&2; exit 1; }
    echo "ParaView 6.0.1 已补齐图形运行库：${runtime_dir}"
    exit 0
fi

actual_size="$(stat -c '%s' -- "${archive_path}")"
if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "安装包大小错误：期望 ${expected_size}，实际 ${actual_size}" >&2
    exit 1
fi
actual_md5="$(md5sum -- "${archive_path}")"
actual_md5="${actual_md5%% *}"
if [[ "${actual_md5}" != "${expected_md5}" ]]; then
    echo "安装包 MD5 与 Kitware 官方清单不匹配。" >&2
    echo "期望 ${expected_md5}，实际 ${actual_md5}" >&2
    exit 1
fi
actual_sha256="$(sha256sum -- "${archive_path}")"
actual_sha256="${actual_sha256%% *}"
if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    echo "ParaView 安装包的 SHA-256 不匹配。" >&2
    echo "期望 ${expected_sha256}，实际 ${actual_sha256}" >&2
    exit 1
fi

if ! tar -tzf "${archive_path}" | awk -v root="${archive_root}/" '
    index($0, root) != 1 { bad = 1 }
    END { exit bad }
'; then
    echo "安装包包含预期顶层目录之外的路径，拒绝解压。" >&2
    exit 1
fi

runtime_parent="$(dirname -- "${runtime_dir}")"
mkdir -p -- "${runtime_parent}"
staging_dir="$(mktemp -d "${runtime_parent}/.paraview-6.0.1.install.XXXXXX")"
cleanup_staging() {
    if [[ -n "${staging_dir:-}" && -d "${staging_dir}" ]]; then
        rm -rf -- "${staging_dir}"
    fi
}
trap cleanup_staging EXIT

echo "正在解压 ParaView 6.0.1..."
tar -xzf "${archive_path}" --strip-components=1 -C "${staging_dir}"
dpkg-deb --extract "${xcb_package_path}" "${staging_dir}"
if [[ ! -x "${staging_dir}/bin/paraview" || ! -x "${staging_dir}/bin/pvpython" ]]; then
    echo "解压后缺少 ParaView 核心程序，安装中止。" >&2
    exit 1
fi

mv -- "${staging_dir}" "${runtime_dir}"
staging_dir=""
trap - EXIT

echo "ParaView 6.0.1 已安装到：${runtime_dir}"
echo "启动命令：${script_dir}/launch_paraview.sh"
