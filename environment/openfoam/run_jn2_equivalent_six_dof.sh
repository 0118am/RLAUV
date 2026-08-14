#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case_root="${CASE_ROOT:-${script_dir}/cases_jn2_equivalent_six_dof_level6_v2}"
resolved_case_settings="${CAMPAIGN_CONFIG:-${script_dir}/experiment_configs/jn2_equivalent_six_dof_level6.json}"

exec 9>"${case_root}/.campaign.lock"
if ! flock -n 9; then
    echo "OpenFOAM campaign is already running for ${case_root}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${script_dir}/env.sh"

# Keep generated cases synchronized with the reviewed campaign configuration
# without rebuilding their mesh or initial fields.
max_co="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_co"])' "${resolved_case_settings}")"
outer_correctors="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pimple_outer_correctors"])' "${resolved_case_settings}")"
force_interval="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["force_execute_interval"])' "${resolved_case_settings}")"

# Establish the common towing-flow initial condition once.  Every oscillatory
# case starts from this same field, matching the already-towing PMM experiment.
baseline="${case_root}/baseline"
steady_marker="${baseline}/.steady_completed"
steady_time_file="${baseline}/.steady_time"
# A reduced campaign may share the exact geometry, mesh, fluid properties and
# towing speed with a previously converged campaign.  Import only the five
# cell fields needed by pimpleFoam; motion fields remain those generated for
# this campaign.
steady_baseline_source="${STEADY_BASELINE_SOURCE:-}"
if [[ ! -f "${steady_marker}" && -n "${steady_baseline_source}" ]]; then
    source_time_file="${steady_baseline_source}/.steady_time"
    if [[ ! -f "${steady_baseline_source}/.steady_completed" || \
          ! -s "${source_time_file}" ]]; then
        echo "shared steady baseline has no completion evidence: ${steady_baseline_source}" >&2
        exit 1
    fi
    source_time="$(<"${source_time_file}")"
    if [[ "$(readlink -f -- "${baseline}/constant/polyMesh")" != \
          "$(readlink -f -- "${steady_baseline_source}/constant/polyMesh")" ]]; then
        echo "shared steady baseline does not use the same polyMesh" >&2
        exit 1
    fi
    mkdir -p -- "${baseline}/${source_time}"
    for field in U p k omega nut; do
        cp -- "${steady_baseline_source}/${source_time}/${field}" \
            "${baseline}/${source_time}/${field}"
        foamDictionary "${baseline}/${source_time}/${field}" \
            -entry FoamFile.location -set "${source_time}" >/dev/null 2>&1
    done
    printf '%s\n' "${source_time}" >"${steady_time_file}"
    : >"${steady_marker}"
fi
if [[ -f "${steady_marker}" && ! -s "${steady_time_file}" ]]; then
    rm -f -- "${steady_marker}"
fi
if [[ ! -f "${steady_marker}" ]]; then
    cp "${script_dir}/steady_initial/system/fvSchemes" "${baseline}/system/fvSchemes"
    cp "${script_dir}/steady_initial/system/fvSolution" "${baseline}/system/fvSolution"
    cp "${script_dir}/steady_initial/system/controlDict" "${baseline}/system/controlDict"

    # Remove generated decomposition directories under the exact baseline case.
    find "${baseline}" -mindepth 1 -maxdepth 1 -type d \
        -name 'processor[0-9]*' -exec rm -rf -- {} +

    foamDictionary "${baseline}/system/decomposeParDict" \
        -entry numberOfSubdomains -set 16 >"${baseline}/log.foamDictionary.steady" 2>&1
    decomposePar -force -case "${baseline}" >"${baseline}/log.decomposePar.steady" 2>&1
    taskset -c 0-15 mpirun -np 16 --map-by core --bind-to core \
        simpleFoam -parallel -case "${baseline}" >"${baseline}/log.simpleFoam" 2>&1
    steady_time="$(foamListTimes -processor -case "${baseline}" -latestTime)"
    if [[ -z "${steady_time}" || "${steady_time}" == "0" ]] || \
       [[ ! -d "${baseline}/processor0/${steady_time}" ]] || \
       ! grep -q 'SIMPLE solution converged in' "${baseline}/log.simpleFoam"; then
        echo "steady initialization did not converge and write complete rank fields" >&2
        exit 1
    fi
    reconstructPar -case "${baseline}" -latestTime >"${baseline}/log.reconstructPar.steady" 2>&1
    if [[ ! -d "${baseline}/${steady_time}" ]]; then
        echo "steady initialization did not reconstruct iteration ${steady_time}" >&2
        exit 1
    fi
    printf '%s\n' "${steady_time}" >"${steady_time_file}"
    : >"${steady_marker}"
fi
steady_time="$(<"${steady_time_file}")"

for motion_file in "${case_root}"/*/motion.json; do
    case_dir="$(dirname -- "${motion_file}")"
    if [[ "$(basename -- "${case_dir}")" == baseline ]]; then
        continue
    fi
    # An interrupted pimpleFoam run restarts from time zero.  Remove only its
    # generated parallel partitions and force history first, otherwise a new
    # run could append to partial output from an older solver configuration.
    if [[ ! -f "${case_dir}/.completed" ]]; then
        find "${case_dir}" -mindepth 1 -maxdepth 1 -type d \
            -name 'processor[0-9]*' -exec rm -rf -- {} +
        rm -rf -- "${case_dir}/postProcessing"
    fi
    foamDictionary "${case_dir}/system/controlDict" \
        -entry maxCo -set "${max_co}" >/dev/null 2>&1
    foamDictionary "${case_dir}/system/controlDict" \
        -entry functions.forces.executeInterval -set "${force_interval}" >/dev/null 2>&1
    foamDictionary "${case_dir}/system/controlDict" \
        -entry functions.forces.writeInterval -set "${force_interval}" >/dev/null 2>&1
    foamDictionary "${case_dir}/system/fvSolution" \
        -entry PIMPLE.nOuterCorrectors -set "${outer_correctors}" >/dev/null 2>&1
    python3 - "${motion_file}" "${max_co}" "${outer_correctors}" "${force_interval}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
data["max_co"] = float(sys.argv[2])
data["pimple_outer_correctors"] = int(sys.argv[3])
data["force_execute_interval"] = int(sys.argv[4])
path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
PY
    # Preserve the generated oscillating point boundary.  mapFields maps the
    # cell-centred towing solution and renames this target-only point field.
    if [[ -f "${case_dir}/0/pointDisplacement.unmapped" && \
          ! -f "${case_dir}/0/pointDisplacement" ]]; then
        mv -- "${case_dir}/0/pointDisplacement.unmapped" \
            "${case_dir}/0/pointDisplacement"
    fi
    if [[ -f "${case_dir}/.mapped_from_steady" ]]; then
        continue
    fi
    # All cases resolve to the exact same polyMesh, so this is an exact
    # one-cell-to-one-cell transfer.  Avoid mapFields interpolation, which is
    # unnecessary here and renames the target-only pointDisplacement field.
    for field in U p k omega nut; do
        cp -- "${baseline}/${steady_time}/${field}" "${case_dir}/0/${field}"
        foamDictionary "${case_dir}/0/${field}" \
            -entry FoamFile.location -set 0 >/dev/null 2>&1
    done
    if [[ ! -f "${case_dir}/0/pointDisplacement" ]]; then
        echo "initialization removed the prescribed point motion in ${case_dir}" >&2
        exit 1
    fi
    : >"${case_dir}/.mapped_from_steady"
done

# Two independent 16-rank jobs occupy the 32 visible CPU cores until all
# generated motions finish.  Exclude the initialization-only baseline.
python3 "${script_dir}/run_cases.py" \
    --cases-dir "${case_root}" \
    --only 'u_*' --only 'v_*' --only 'w_*' \
    --only 'p_*' --only 'q_*' --only 'r_*' \
    --np 16 \
    --jobs 2 \
    --resume \
    --bind-to-core \
    --cpu-set 0-15 \
    --cpu-set 16-31

if [[ -n "${RESULTS_OUTPUT:-}" ]]; then
    python3 -m environment.openfoam.analysis \
        --cases-root "${case_root}" \
        --config "${resolved_case_settings}" \
        --output-dir "${RESULTS_OUTPUT}"
fi
