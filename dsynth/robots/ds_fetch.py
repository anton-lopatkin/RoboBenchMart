from copy import deepcopy
from transforms3d import euler
import numpy as np
import sapien
from mani_skill.utils import sapien_utils
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils.structs.link import Link
from dsynth import PACKAGE_DIR

@register_agent()
class DSFetchBasket(Fetch):
    uid = "ds_fetch_basket"
    urdf_path = f"{PACKAGE_DIR}/assets/urdf/fetch/fetch_basket.urdf"
    urdf_arm_ik_path = f"{PACKAGE_DIR}/assets/urdf/fetch/fetch_torso_up.urdf"

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([0, 0, 0, 0.386, 0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0, np.pi / 3, 0, 0.015, 0.015,]),  # fmt: skip
        )
    )
    
    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="fetch_hand",
                pose=Pose.create_from_pq([0.1, 0, -0.1], euler.euler2quat(np.pi, -np.pi / 2, 0)),
                width=1200,
                height=1200,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="gripper_link",
            ),
            CameraConfig(
                uid="left_base_camera_link",
                pose=Pose.create_from_pq([-0.5, 0.5, 0], euler.euler2quat(0, 0.3, -0.2)),
                width=1200,
                height=1200,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="head_camera_link",
            ),
            CameraConfig(
                uid="right_base_camera_link",
                pose=Pose.create_from_pq([-0.5, -0.5, 0], euler.euler2quat(0, 0.3, 0.2)),
                width=1200,
                height=1200,
                fov=1.5,
                near=0.01,
                far=100,
                entity_uid="head_camera_link",
            ),
        ]
    @property
    def _controller_configs(self):
        controller_configs = super()._controller_configs

        body_pd_joint_delta_target_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            -0.1,
            0.1,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=True,
            use_target=True
        )

        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=False,
            normalize_action=False,
        )

        # useful to keep body unmoving from passed position
        stiff_body_pd_joint_target_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
            use_target=True
        )

        controller_configs['pd_joint_pos']['body'] = body_pd_joint_pos

        return deepcopy_dict(controller_configs)

    def _after_init(self):
        super()._after_init()
        self.shoulder_pan_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "shoulder_pan_link"
        )
