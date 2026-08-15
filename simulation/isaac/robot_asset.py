import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg

from robot.assets.isaac import T60_USD_PATH
from robot.dynamics.parameters import AUV

USD_PATH = str(T60_USD_PATH)

AUV_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        mass_props=sim_utils.MassPropertiesCfg(
            mass=AUV.mass_kg,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5),
    )
)
"""Spawn configuration for the AUV validation vehicle."""
