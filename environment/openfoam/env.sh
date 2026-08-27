#!/usr/bin/env bash
# Source this file to load the OpenCFD OpenFOAM v2512 environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请用 source environment/openfoam/env.sh 加载环境，而不是直接执行。" >&2
    exit 2
fi

if command -v pimpleFoam >/dev/null 2>&1 && [[ "${FOAM_API:-}" == "2512" ]]; then
    return 0
fi

_auv_openfoam_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_auv_openfoam_local_root="${AUV_OPENFOAM_ROOT:-${_auv_openfoam_script_dir}/.runtime/openfoam2512}"
_auv_openfoam_local_bashrc="${_auv_openfoam_local_root}/usr/lib/openfoam/openfoam2512/etc/bashrc"
_auv_openfoam_candidates=()
if [[ -n "${AUV_OPENFOAM_BASHRC:-}" ]]; then
    _auv_openfoam_candidates+=("${AUV_OPENFOAM_BASHRC}")
fi
_auv_openfoam_candidates+=(
    "${_auv_openfoam_local_bashrc}"
    "/usr/lib/openfoam/openfoam2512/etc/bashrc"
    "/opt/openfoam2512/etc/bashrc"
    "/opt/OpenFOAM-v2512/etc/bashrc"
)

_auv_openfoam_loaded=false
for _auv_openfoam_bashrc in "${_auv_openfoam_candidates[@]}"; do
    if [[ -f "${_auv_openfoam_bashrc}" ]]; then
        if [[ "${_auv_openfoam_bashrc}" == "${_auv_openfoam_local_bashrc}" ]]; then
            _auv_openfoam_local_lib="${_auv_openfoam_local_root}/usr/lib/x86_64-linux-gnu"
            if [[ -d "${_auv_openfoam_local_lib}" ]]; then
                export LD_LIBRARY_PATH="${_auv_openfoam_local_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            fi
        fi
        _auv_openfoam_restore_nounset=false
        if [[ "$-" == *u* ]]; then
            set +u
            _auv_openfoam_restore_nounset=true
        fi
        # OpenFOAM's bashrc parses its positional parameters as configuration
        # fragments.  When env.sh is sourced by launch_openfoam.sh, the
        # launcher's command and filenames would otherwise be interpreted (and
        # potentially sourced) before the command is executed.
        _auv_openfoam_saved_args=("$@")
        set --
        # shellcheck disable=SC1090
        if source "${_auv_openfoam_bashrc}"; then
            _auv_openfoam_source_status=0
        else
            _auv_openfoam_source_status=$?
        fi
        set -- "${_auv_openfoam_saved_args[@]}"
        if [[ "${_auv_openfoam_restore_nounset}" == true ]]; then
            set -u
        fi
        if (( _auv_openfoam_source_status != 0 )); then
            continue
        fi
        _auv_openfoam_user_platform="${_auv_openfoam_script_dir}/.runtime/user/platforms/${WM_OPTIONS}"
        export FOAM_USER_APPBIN="${_auv_openfoam_user_platform}/bin"
        export FOAM_USER_LIBBIN="${_auv_openfoam_user_platform}/lib"
        mkdir -p -- "${FOAM_USER_APPBIN}" "${FOAM_USER_LIBBIN}"
        export PATH="${_auv_openfoam_script_dir}/bin:${PATH}"
        export PATH="${FOAM_USER_APPBIN}:${PATH}"
        export LD_LIBRARY_PATH="${FOAM_USER_LIBBIN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        _auv_openfoam_loaded=true
        break
    fi
done

unset _auv_openfoam_candidates _auv_openfoam_bashrc _auv_openfoam_local_bashrc
unset _auv_openfoam_local_lib _auv_openfoam_local_root _auv_openfoam_script_dir
unset _auv_openfoam_user_platform
unset _auv_openfoam_restore_nounset _auv_openfoam_source_status
unset _auv_openfoam_saved_args

if [[ "${_auv_openfoam_loaded}" != true ]] || ! command -v pimpleFoam >/dev/null 2>&1; then
    unset _auv_openfoam_loaded
    echo "未找到 OpenCFD OpenFOAM v2512。设置 AUV_OPENFOAM_BASHRC 后重新 source 此文件。" >&2
    return 1
fi
unset _auv_openfoam_loaded

if [[ "${FOAM_API:-}" != "2512" ]]; then
    echo "检测到 FOAM_API=${FOAM_API:-unknown}，本部署锁定 OpenCFD v2512，拒绝混用发行版语法。" >&2
    return 1
fi
