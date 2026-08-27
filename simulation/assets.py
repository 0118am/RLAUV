"""Isaac asset configuration derived from robot-owned data."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from robot.assets.isaac import T60_USD_PATH
from robot.dynamics.parameters import AUV


T60_ASSET_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(T60_USD_PATH),
        mass_props=sim_utils.MassPropertiesCfg(mass=AUV.mass_kg),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        copy_from_source=False,
    ),
)
