"""Isaac Lab scene construction around NVIDIA's pinned SO-101 workshop.

This module must only be imported after :class:`isaaclab.app.AppLauncher` starts
Isaac Sim. Keeping it separate lets all configuration/writer tests run locally.
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass

from sim_to_real_so101.assets.so101 import S0101_NO_CAMERA_CFG
from sim_to_real_so101.tasks.task_env_cfg import camera_object
from sim_to_real_so101.tasks.vials_to_rack_env_cfg import (
    VialsToRackDREnvCfg,
    VialsToRackDRSceneCfg,
    VialsToRackEventDRCfg,
    VialsToRackObservationsCfg,
)

from .config import Stage1Config
from .worlds import WorldProfile


@configclass
class Stage1SceneCfg(VialsToRackDRSceneCfg):
    """One SO-101, one rear stand camera, one gripper IMU per vector environment."""

    # Use NVIDIA's camera-free SO-101 USD and re-enable its contact reporters.
    # This removes the upstream wrist-camera prim, not merely its observation term.
    robot: ArticulationCfg = S0101_NO_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True

    camera_ego = None
    camera_external_D455 = None

    rear_camera = camera_object.replace()
    rear_camera.prim_path = (
        "{ENV_REGEX_NS}/LightStudio/LightBox/camera_mount/rsd455/RSD455/"
        "Camera_OmniVision_OV9782_Right"
    )
    rear_camera.spawn = None

    gripper_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        update_period=0.0,
        gravity_bias=(0.0, 0.0, 9.81),
        debug_vis=False,
    )

    # All robot bodies are observed. The recorder excludes expected base and
    # jaw/gripper contacts when deriving collision labels. The workshop's
    # filtered jaw sensor separately gives exact per-vial forces.
    robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        debug_vis=False,
    )


@configclass
class Stage1EventCfg(VialsToRackEventDRCfg):
    # It references the wrist camera removed above.
    reset_camera_ego_fov = None


@configclass
class Stage1ObservationsCfg(VialsToRackObservationsCfg):
    # Sensor tensors are read directly, preserving native values and avoiding a
    # second copy. The official subtask/contact observation terms remain active.
    visual = None


@configclass
class Stage1EnvCfg(VialsToRackDREnvCfg):
    scene: Stage1SceneCfg = Stage1SceneCfg()
    events: Stage1EventCfg = Stage1EventCfg()
    observations: Stage1ObservationsCfg = Stage1ObservationsCfg()


def _required_randomizer(name: str):
    function = getattr(isaac_mdp, name, None)
    if function is None:
        raise RuntimeError(f"Isaac Lab is missing required domain randomizer: {name}")
    return function


def _install_physics_randomization(cfg: Stage1EnvCfg, profile: WorldProfile) -> None:
    cfg.events.stage1_actuator_gains = EventTerm(
        func=_required_randomizer("randomize_actuator_gains"),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": profile.actuator_gain_scale,
            "damping_distribution_params": profile.actuator_gain_scale,
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    cfg.events.stage1_joint_friction = EventTerm(
        func=_required_randomizer("randomize_joint_parameters"),
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": profile.joint_friction_scale,
            "armature_distribution_params": profile.joint_friction_scale,
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    for object_name in ("vial_1", "vial_2", "vial_3"):
        setattr(
            cfg.events,
            f"stage1_{object_name}_mass",
            EventTerm(
                func=_required_randomizer("randomize_rigid_body_mass"),
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(object_name),
                    "mass_distribution_params": profile.object_mass_scale,
                    "operation": "scale",
                    "distribution": "uniform",
                },
            ),
        )
        setattr(
            cfg.events,
            f"stage1_{object_name}_material",
            EventTerm(
                func=_required_randomizer("randomize_rigid_body_material"),
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(object_name),
                    "static_friction_range": profile.object_friction_range,
                    "dynamic_friction_range": profile.object_friction_range,
                    "restitution_range": (0.0, 0.08),
                    "num_buckets": 64,
                    "make_consistent": True,
                },
            ),
        )


def build_env_cfg(
    config: Stage1Config,
    profile: WorldProfile,
    num_envs: int,
    record_segmentation: bool,
    seed: int,
) -> Stage1EnvCfg:
    """Create a configured vector world; called once per generation job."""

    cfg = Stage1EnvCfg()
    # The workshop base's post-init intentionally forces teleoperation to one
    # environment. Override it only after construction for vector generation.
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = config.runtime.device
    cfg.sim.dt = 1.0 / config.runtime.physics_hz
    cfg.decimation = config.runtime.decimation
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = 24.0 * 60.0 * 60.0
    cfg.rerender_on_reset = True

    camera = cfg.scene.rear_camera
    camera.width = config.camera.width
    camera.height = config.camera.height
    camera.update_period = 1.0 / config.camera.fps
    camera.data_types = ["rgb"]
    camera.colorize_instance_segmentation = False
    if record_segmentation:
        camera.data_types.append("instance_id_segmentation_fast")

    cfg.scene.gripper_imu.prim_path = config.imu.prim_path
    cfg.scene.gripper_imu.gravity_bias = (
        (0.0, 0.0, 9.81) if config.imu.include_gravity else (0.0, 0.0, 0.0)
    )

    reset = cfg.events.reset_vials_setup.params
    reset["rack_pose_range"] = profile.rack_pose_range
    reset["pose_range"] = profile.vial_pose_range
    reset["rack_placement_prob"] = profile.rack_placement_probability
    cfg.events.reset_robot_position.params["position_range"] = (
        -profile.robot_start_offset_rad,
        profile.robot_start_offset_rad,
    )
    cfg.events.reset_lightbox_light_exposure.params["exposure_range"] = profile.light_exposure_range
    cfg.events.reset_camera_external_pose.params["pos_range"] = {
        axis: (-profile.camera_position_jitter_m, profile.camera_position_jitter_m)
        for axis in ("x", "y", "z")
    }
    cfg.events.reset_camera_external_pose.params["rot_range"] = {
        axis: (-profile.camera_rotation_jitter_rad, profile.camera_rotation_jitter_rad)
        for axis in ("roll", "pitch", "yaw")
    }

    # Keep the global dome light in every profile. Besides supplying visual
    # diversity, it prevents per-environment direct-light differences from
    # silently producing dark tiled-camera frames; live preflight checks every
    # environment's pixels as a second guard.
    _install_physics_randomization(cfg, profile)
    return cfg
