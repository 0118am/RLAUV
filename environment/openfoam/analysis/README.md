# OpenFOAM hydrodynamic matrix analysis

Run the fitter after all prescribed-motion cases have produced OpenCFD v2512
`postProcessing/forces/**/{force.dat,moment.dat}`:

```bash
python3 -m openfoam.analysis \
  --cases-root environment/openfoam/cases \
  --output-dir environment/openfoam/results
```

The equivalent deployment-friendly entry point is
`python3 environment/openfoam/analysis/fit_matrices.py --cases-root environment/openfoam/cases`.

Each oscillatory case must contain `motion.json` plus OpenCFD v2512
`postProcessing/forces/**/{force.dat,moment.dat}` output. Legacy combined
`forces.dat` is also accepted. The canonical motion fields are
`dof`, `dof_index`, `motion_kind`, `axis`, `amplitude_si`, `omega_rad_s`,
`phase_rad`, `settle_cycles`, `sample_cycles`, `cofr_global_m`, and
`com_initial_global_m`. Generator aliases such as `kind`, `amplitude_m`,
`amplitude_rad`, `amplitude_deg`, and `centre_of_rotation_m` are accepted.
Baseline/rest cases are skipped.

The output convention is body FLU at the moving COM, with DOFs
`[u,v,w,p,q,r]`, wrench `[X,Y,Z,K,M,N]`, and fluid-on-body model

```text
tau = -M_A*nudot - C_A(nu,M_A)*nu
      - D_L*nu - D_Q*(abs(nu)*nu)
```

The primary estimate first resamples every complete cycle onto the same
uniform phase grid (`phase_samples_per_cycle`, default `256`), then pairs
samples half a period apart. Equal phase rows per cycle prevent adaptive time
steps or denser cases from becoming accidental regression weights. The pairing
cancels steady bias and the even-in-velocity added-mass Coriolis load without
discarding any off-axis wrench response. Requested partial cycles are rejected
rather than silently underweighted. `config_updates.json` contains the project keys
`added_mass_diag`, `linear_damping`, and `quadratic_damping`, each as a full
6x6 matrix.
