# AUV/T60 workflows

Supported entry points:

- `train/`: trajectory training and competence-gated curriculum supervision.
- `evaluate/`: policy evaluation, disturbance matrices, and plotting.
- `export/`: deployable policy export.
- `replay/`: measured-action replay and validation.

`common/` contains shared command builders and experiment helpers. Calibration
and fitting commands live in `environment/calibration/`.

Workflow scripts may parse arguments, discover experiment files, and write
reports, but must delegate physics, fitting, profiles, and validation metrics
to `simulation/isaac/envs/auv/`, `environment/`, or `robot/` according to
ownership. New reusable math must not be added directly to a workflow.
