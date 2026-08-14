# Training

Training requires the project Conda environment:

```bash
conda activate env_isaaclab
cd <IsaacLab_Path>
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/isaac-auv-env/simulation/isaac/workflows/train/trajectory.py \
  --task Isaac-AUV-Traj-Direct-v1 \
  --num_envs 2048
```

Use `trajectory.ipynb` to select the architecture, reward profile, simulation
profile, Domain Randomization recipe, curriculum, and run parameters.
`competence_curriculum.py` supervises segmented train/evaluate gates.

