import torch
import numpy as np
import pandas as pd
import sapien
from mani_skill.utils.registration import register_env
from dsynth.envs.darkstore_cont_base import DarkstoreContinuousBaseEnv
from mani_skill.utils.structs.pose import Pose


@register_env('CollectOrderItemsContEnv', max_episode_steps=200000)
class CollectOrderItemsContEnv(DarkstoreContinuousBaseEnv):
    TARGET_PRODUCT_NAME = None
    ROBOT_INIT_POSE_RANDOM_ENABLED = True

    TARGET_POS_THRESH = 0.2
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        self.target_sizes = np.array([self.TARGET_POS_THRESH, self.TARGET_POS_THRESH, self.TARGET_POS_THRESH])
    
    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.target_products_df = None
        
        if self.markers_enabled:
            target_markers_iterator = {key: iter(val) for key, val in self.target_markers.items()}

        self.target_product_names = {idx: self.TARGET_PRODUCT_NAME for idx in range(self.num_envs)}

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_prducts_df = self.products_df[self.products_df['scene_idx'] == scene_idx]
            
            if self.TARGET_PRODUCT_NAME is None:
                product_name = self._batched_episode_rng[scene_idx].choice(sorted(scene_prducts_df['product_name'].unique()))
                self.target_product_names[scene_idx] = product_name
        
            else:
                product_name = self.TARGET_PRODUCT_NAME
                if not self.TARGET_PRODUCT_NAME in scene_prducts_df['product_name'].unique():
                    raise RuntimeError(f"Product {self.TARGET_PRODUCT_NAME} is not present on scene #{scene_idx}")
            
            if self.target_products_df is None:
                self.target_products_df = scene_prducts_df[scene_prducts_df['product_name'] == product_name]
            else:
                self.target_products_df = pd.concat([self.target_products_df,
                    scene_prducts_df[scene_prducts_df['product_name'] == product_name]
                                                    ])
            
            if self.markers_enabled:
                target_products = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
                for actor_name in target_products['actor_name']:
                    actor = self.actors['products'][actor_name]
                    try:
                        target_marker = next(target_markers_iterator[scene_idx])
                    except StopIteration:
                        raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}")
                    target_marker.set_pose(actor.pose)

    def _compute_robot_init_pose(self, env_idx = None):
        robot_origins, robot_angles, directions_to_shelf = super()._compute_robot_init_pose(env_idx)
        for idx in env_idx:
            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                batched_rng = self._batched_episode_rng
                if self.extra_robot_pose_randomization:
                    batched_rng = self._batched_init_pose_rng
                idx = idx.cpu().item()
                direction_to_shelf = directions_to_shelf[idx]
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

                delta_par = batched_rng[idx].rand() * 0.2
                delta_perp = (batched_rng[idx].rand() - 0.5) * 0.5

                robot_origins[idx] += -direction_to_shelf * delta_par + perp_direction * delta_perp
                robot_angles[idx] += (batched_rng[idx].rand() - 0.5) * np.pi / 4

        return robot_origins, robot_angles, directions_to_shelf
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if self.robot_uids in ["fetch", "ds_fetch", "ds_fetch_basket"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.34,
                    0,
                    0,
                    0,
                    1.4,
                    0,
                    0.76,
                    0,
                    - 2 * np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)

    def evaluate(self):
        target_pos = self.calc_target_pose().p 
        # target_pos[:, 2] -= self.target_sizes[2] / 2
        # tolerance = torch.tensor(self.target_sizes / 2, dtype=torch.float32).to(self.device)
        tolerance = torch.tensor([self.TARGET_POS_THRESH, self.TARGET_POS_THRESH, self.TARGET_POS_THRESH]).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                target_product_pos = self.actors['products'][actor_name].pose.p
                scene_is_obj_placed = torch.all(
                    (target_product_pos >= (target_pos[scene_idx] - tolerance)) & 
                    (target_product_pos <= (target_pos[scene_idx] + tolerance)),
                    dim=-1
                )
                if scene_is_obj_placed:
                    break
            
            is_obj_placed.append(scene_is_obj_placed)

        is_obj_placed = torch.cat(is_obj_placed)
        
        is_robot_static = self.agent.is_static(0.2)

        is_non_target_products_replaced = torch.zeros_like(is_robot_static, dtype=bool)

        for scene_idx in range(self.num_envs):
            scene_products_df = self.products_df[self.products_df['scene_idx'] == scene_idx]

            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            non_target_actors = set(scene_products_df['actor_name']) - set(scene_target_products_df['actor_name'])
            
            for actor_name in non_target_actors:
                actor = self.actors['products'][actor_name]
                if actor_name in self.products_initial_poses:
                    if not torch.all(torch.isclose(actor.pose.raw_pose, self.products_initial_poses[actor_name], rtol=0.1, atol=0.1)):
                        is_non_target_products_replaced[scene_idx] = True

                        if self.markers_enabled:
                            # make marker red if non-target product moved
                            render_component = self.target_volumes[scene_idx][0]._objs[0].find_component_by_type(
                                sapien.pysapien.render.RenderBodyComponent
                            )
                            render_component.render_shapes[0].material.base_color = [1.0, 0.0, 0.0, 0.5]

                        break


        return {
            "is_obj_placed" : is_obj_placed,
            "is_robot_static" : is_robot_static,
            "is_non_target_produncts_displaced" : is_non_target_products_replaced,
            "success": is_obj_placed & is_robot_static & (~is_non_target_products_replaced),
        }


    def pick_item(self, item_id):
        item = [
            item for item in self.actors['products'].values()
            if item.per_scene_id[0].item() == item_id
        ][0]

        item.set_pose(Pose.create_from_pq(p=self.agent.tcp_pos))
        self.grasped_item_id = item_id


    def place_to_basket(self):
        item = [
            item for item in self.actors['products'].values()
            if item.per_scene_id[0].item() == self.grasped_item_id
        ][0]
        robot_pose = self.agent.base_link.pose
        basket_shift = Pose.create_from_pq(p=[[0.3, 0.25, 0.14]])
        basket = robot_pose * basket_shift 
        item.set_pose(Pose.create_from_pq(p=basket.p))
        self.grasped_item_id = None



    def calc_target_pose(self):
        robot_pose = self.agent.base_link.pose
        basket_shift = Pose.create_from_pq(p=[[0.3, 0.25, 0.14]] * self.num_envs)
        return robot_pose * basket_shift 
       
    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'prepare the ingredients for a latte')

    def _after_simulation_step(self):
        #does not work on gpu sim
        if self.markers_enabled:
            target_pose = self.calc_target_pose()
            for scene_idx in range(self.num_envs):
                self.target_volumes[scene_idx][0].set_pose(
                    Pose.create_from_pq(p=target_pose.p[scene_idx],
                                        q=target_pose.q[scene_idx])
                )
            # self.target_volume.set_pose(target_pose)


@register_env('CollectOrderItemsContLatteEnv', max_episode_steps=200000)
class CollectOrderItemsContLatteEnv(CollectOrderItemsContEnv):
    TARGET_PRODUCTS_NAMES = ['Coffee', 'Milk']
