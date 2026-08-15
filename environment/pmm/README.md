# PMM identification

Raw towing-tank CSV records, the six-DOF identification configuration, and
frequency-resolved fit outputs live together here. Run from the repository
root with:

```bash
python environment/pmm/six_dof_identification.py
```

The script resolves its default inputs and outputs relative to this directory.
It is a compatibility entry point over focused modules:

- `pmm_config.py`: configuration and raw-record preflight;
- `pmm_kinematics.py`: Fourier reconstruction and frame transforms;
- `pmm_trials.py`: one-trial reconstruction and harmonic projection;
- `pmm_fitting.py`: robust derivative fitting and matrix assembly;
- `pmm_reporting.py`: tables, metadata, plots, and reports.

The checked-in `hydro_results/` files are analysis products; refactoring does
not select, overwrite, or republish them automatically.
