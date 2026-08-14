#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CASE_ROOT="${script_dir}/cases_jn2_port_starboard_symmetric_minimal_level6_v1"
export CAMPAIGN_CONFIG="${script_dir}/experiment_configs/jn2_port_starboard_symmetric_minimal_level6.json"
export STEADY_BASELINE_SOURCE="${script_dir}/cases_jn2_equivalent_six_dof_level6_v2/baseline"
export RESULTS_OUTPUT="${script_dir}/results_jn2_port_starboard_symmetric_minimal_level6_v1"
exec "${script_dir}/run_jn2_equivalent_six_dof.sh"
