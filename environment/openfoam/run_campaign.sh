#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--pilot" ) ]]; then
    printf 'Usage: %s [--pilot]\n' "$0" >&2
    exit 2
fi
pilot=0
if [[ ${1:-} == "--pilot" ]]; then
    pilot=1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/../.." && pwd)
config="${script_dir}/config.json"
geometry="${script_dir}/geometry/validated_locked_rotor_v1/wetted_body_m.obj"
repair_report="${script_dir}/geometry/validated_locked_rotor_v1/selection_report.json"
case_store="${script_dir}/cases"
result_store="${script_dir}/results"
cfd_np=${AUV_CFD_NP:-8}
cfd_jobs=${AUV_CFD_JOBS:-1}

# shellcheck disable=SC1091
source "${script_dir}/env.sh"
cd "${repository_root}"

wmkdepend="${WM_PROJECT_DIR}/platforms/tools/${WM_ARCH}${WM_COMPILER}/wmkdepend"
if [[ ! -x "${wmkdepend}" ]]; then
    make -C "${WM_PROJECT_DIR}/wmake/src"
fi
(
    cd "${script_dir}/boundary_conditions/rampedAuvMotion"
    wmake libso
)

case_root="${case_store}/current"
mesh_arguments=(
    "${geometry}"
    --repair-report "${repair_report}"
    --config "${config}"
    --cases-dir "${case_root}"
)

if [[ -e "${case_root}" ]] && python3 "${script_dir}/build_mesh.py" \
    "${mesh_arguments[@]}" --verify-existing; then
    printf 'Reuse shared mesh: %s\n' "${case_root}"
elif [[ -e "${case_root}" ]]; then
    printf 'Rebuild incomplete campaign: %s\n' "${case_root}" >&2
    python3 "${script_dir}/build_mesh.py" "${mesh_arguments[@]}" --force
else
    python3 "${script_dir}/build_mesh.py" "${mesh_arguments[@]}"
fi

run_arguments=(
    --cases-dir "${case_root}"
    --config "${config}"
    --np "${cfd_np}"
    --jobs "${cfd_jobs}"
    --resume
)
if [[ -n "${AUV_CFD_CPU_SETS:-}" ]]; then
    run_arguments+=(--bind-to-core)
    IFS=';' read -r -a cfd_cpu_sets <<< "${AUV_CFD_CPU_SETS}"
    for cfd_cpu_set in "${cfd_cpu_sets[@]}"; do
        run_arguments+=(--cpu-set "${cfd_cpu_set}")
    done
fi
if [[ "${AUV_CFD_RECONSTRUCT:-0}" == "1" ]]; then
    run_arguments+=(--reconstruct)
fi
if (( pilot )); then
    # Reuse these representative completions in the subsequent 24-case run.
    run_arguments+=(
        --only steady_damping_v_pos_0p400mps
        --only oscillatory_damping_q_rate0p800radps_f1p000hz
        --only added_mass_u_vel0p040mps_f1p000hz
        --only added_mass_r_rate0p080radps_f1p000hz
    )
fi
python3 "${script_dir}/run_cases.py" "${run_arguments[@]}"

if (( pilot )); then
    printf 'Pilot completed: %s\n' "${case_root}"
    printf 'Resume the campaign with: %s\n' "${script_dir}/run_campaign.sh"
    exit 0
fi

python3 -m environment.openfoam.analysis \
    --cases-root "${case_root}" \
    --config "${config}" \
    --output-root "${result_store}"

printf 'Cases: %s\n' "${case_root}"
printf 'Install the selected hydrodynamic_fit.json with publish_results.py.\n'
