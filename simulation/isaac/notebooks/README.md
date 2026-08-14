# Experiment notebooks

- `train.ipynb` is the human-facing training recipe and launch surface.
- `evaluate.ipynb` selects checkpoints, runs evaluation matrices, and plots results.

Reusable logic belongs in `simulation/isaac/workflows/`; notebooks should
remain thin configuration and reporting layers.
