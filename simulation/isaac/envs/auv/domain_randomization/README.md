# Domain-randomization feature groups

This folder is the single runtime home for selectable domain-randomization
features.  Versioned numeric ranges remain in `simulation/isaac/configs/domain_randomization/*.json` and
their schema lives in `environment/profiles/domain_randomization.py`.

| Feature | Runtime module | Includes |
| --- | --- | --- |
| `rigid_body` | `rigid_body.py` | payload ensemble, mass, volume, inertia, COM/COB |
| `current` | `current.py` | reset current, current direction, smooth time variation |
| `hydrodynamics` | `hydrodynamics.py` | damping, speed-curve scale, added mass |
| `actuators` | `actuators.py` | thrust scale, tau, delay, slew limit, quantization, dropout, wake, reaction torque |
| `battery` | `battery.py` | initial voltage and voltage-drop rate |
| `observations` | `observations.py` | policy-observation delay, sampling, filtering, dropout, noise, and drift |

`enabled_features` is optional in a recipe.  Omitting it preserves old recipe
behaviour and enables every feature.  New recipes may pin a subset.  A
`TrainRequest(domain_randomization_features=(...))` overrides that list for
one training run and is written into the resolved Hydra configuration and
effective evaluation configuration.

For example, an actuator-and-voltage ablation is:

```python
TrainRequest(
    reward_profile="policy_0",
    domain_randomization_spec=DOMAIN_RANDOMIZATION_SPEC,
    domain_randomization_features=("actuators", "battery"),
)
```

The corresponding direct Hydra overrides are:

```text
env.domain_randomization_feature_override_enabled=true
env.domain_randomization.enabled_features=["actuators","battery"]
```

The global `use_custom_randomization` gate still controls whether any
randomization is sampled.  An empty explicit feature list is useful for a
recipe-identified deterministic ablation.
