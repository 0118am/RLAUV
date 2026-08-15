# Forced-oscillation convergence variants

The JSON files in `configs/` are complete snapshots of the formal experiment
configuration. They retain the reviewed geometry, motion
matrix, `3 + 5` cycles, four writes per cycle, and `purgeWrite 4`. Only the
convergence variable named by each file changes:

- `mesh_coarse.json`: `72 x 36 x 36` base mesh;
- `mesh_nominal.json`: `96 x 48 x 48` base mesh;
- `mesh_fine.json`: `128 x 64 x 64`, with both snappy cell caps at `24M`;
- `dt800.json`: 800 steps per cycle and `maxCo 0.25`;
- `domain_expanded.json`: `[-4, 6] x [-2.5, 2.5] x [-2.5, 2.5]` with
  `120 x 60 x 60` cells, preserving the nominal base-cell size.

Always give each variant a separate cases directory. In particular, do not use
or force-rebuild `environment/openfoam/cases_locked_rotor_v1`.

After sourcing OpenCFD v2512, the four independent meshes can be built with:

```bash
source environment/openfoam/env.sh

for variant in mesh_coarse mesh_nominal mesh_fine domain_expanded; do
  python3 environment/openfoam/build_mesh.py \
    environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.stl \
    --prepared-input \
    --repair-report environment/openfoam/geometry/validated_locked_rotor_v1/selection_report.json \
    --expected-displaced-volume-m3 0.011304505834 \
    --mesh-volume-relative-tolerance 0.055 \
    --config "environment/openfoam/convergence/configs/${variant}.json" \
    --cases-dir "environment/openfoam/convergence/cases_${variant}" \
    --mesh-only
done
```

Generate the motion cases against their own checked meshes without a baseline:

```bash
for variant in mesh_coarse mesh_nominal mesh_fine domain_expanded; do
  python3 environment/openfoam/generate_cases.py \
    --config "environment/openfoam/convergence/configs/${variant}.json" \
    --output "environment/openfoam/convergence/cases_${variant}" \
    --geometry environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.stl \
    --geometry-mode symlink \
    --base-poly-mesh "environment/openfoam/convergence/cases_${variant}/mesh_case/constant/polyMesh" \
    --poly-mesh-mode symlink \
    --repair-report environment/openfoam/geometry/validated_locked_rotor_v1/selection_report.json \
    --no-baseline
done
```

The time-step variant intentionally reuses the nominal checked mesh:

```bash
python3 environment/openfoam/generate_cases.py \
  --config environment/openfoam/convergence/configs/dt800.json \
  --output environment/openfoam/convergence/cases_dt800 \
  --geometry environment/openfoam/geometry/validated_locked_rotor_v1/wetted_body_m.stl \
  --geometry-mode symlink \
  --base-poly-mesh environment/openfoam/convergence/cases_mesh_nominal/mesh_case/constant/polyMesh \
  --poly-mesh-mode symlink \
  --repair-report environment/openfoam/geometry/validated_locked_rotor_v1/selection_report.json \
  --no-baseline
```

Run only the reviewed representative cases; do not launch all 24 cases for
every convergence variant:

```bash
for variant in mesh_coarse mesh_nominal mesh_fine; do
  python3 environment/openfoam/run_cases.py \
    --cases-dir "environment/openfoam/convergence/cases_${variant}" \
    --np 8 --jobs 1 --resume \
    --only 'v_amp0p025m_f0p75hz' \
    --only 'v_amp0p025m_f1p50hz' \
    --only 'q_amp5p0deg_f1p50hz'
done

python3 environment/openfoam/run_cases.py \
  --cases-dir environment/openfoam/convergence/cases_dt800 \
  --np 8 --jobs 1 --resume \
  --only 'v_amp0p025m_f0p75hz' \
  --only 'v_amp0p025m_f1p50hz' \
  --only 'q_amp5p0deg_f1p50hz'

python3 environment/openfoam/run_cases.py \
  --cases-dir environment/openfoam/convergence/cases_domain_expanded \
  --np 8 --jobs 1 --resume \
  --only 'v_amp0p025m_f0p75hz'
```

The `0.75 Hz` sway case above is the common five-variant case used for the
combined grid/time-step/domain check. Compare it with:

```bash
python3 -m openfoam.convergence.compare \
  --coarse environment/openfoam/convergence/cases_mesh_coarse/v_amp0p025m_f0p75hz \
  --nominal environment/openfoam/convergence/cases_mesh_nominal/v_amp0p025m_f0p75hz \
  --fine environment/openfoam/convergence/cases_mesh_fine/v_amp0p025m_f0p75hz \
  --dt environment/openfoam/convergence/cases_dt800/v_amp0p025m_f0p75hz \
  --domain environment/openfoam/convergence/cases_domain_expanded/v_amp0p025m_f0p75hz \
  --output-dir environment/openfoam/convergence/results/v_amp0p025m_f0p75hz
```

This writes `convergence_report.json` and `convergence_report.md`. The command
requires five distinct case directories with identical motion definitions,
complete requested cycles, finite loads, and no restart gap crossing a sampled
cycle. It reports the excited-DOF
diagonal added mass, peak-speed secant damping
`D_eff = DL + DQ*v_peak`, measured main-load amplitude/phase, and fit residual.
Relative differences use nominal versus refined time step and nominal versus
expanded domain. Three-grid GCI is emitted only for a monotonic sequence with
positive observed order; an oscillatory or divergent sequence is reported
explicitly without a GCI value.

The JSON contains raw comparisons and GCI diagnostics, without embedding a
second set of project-specific pass/fail thresholds. Acceptance belongs to the
experiment review that consumes the report.
