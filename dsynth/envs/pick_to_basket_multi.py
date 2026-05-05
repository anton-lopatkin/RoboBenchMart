import pandas as pd
import sapien
import torch
from mani_skill.utils.registration import register_env

from dsynth.envs import PickToBasketContEnv


@register_env("PickToBasketMultiContEnv", max_episode_steps=200000)
class PickToBasketMultiContEnv(PickToBasketContEnv):
    TARGET_PRODUCTS_NAMES = None

    def setup_target_objects(self, env_idxs):
        self.target_products_df = None

        if self.markers_enabled:
            target_markers_iterator = {
                key: iter(val) for key, val in self.target_markers.items()
            }

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_products_df = self.products_df[
                self.products_df["scene_idx"] == scene_idx
            ]

            if self.TARGET_PRODUCTS_NAMES is None:
                unique_products = sorted(scene_products_df["product_name"].unique())
                self.TARGET_PRODUCTS_NAMES = self._batched_episode_rng[
                    scene_idx
                ].choice(unique_products, size=2, replace=False)

            for product_name in self.TARGET_PRODUCTS_NAMES:
                if product_name not in scene_products_df["product_name"].unique():
                    raise RuntimeError(
                        f"Product {product_name} is not present on scene #{scene_idx}"
                    )

            scene_target_products_df = scene_products_df[
                scene_products_df["product_name"].isin(self.TARGET_PRODUCTS_NAMES)
            ]
            if self.target_products_df is None:
                self.target_products_df = scene_target_products_df
            else:
                self.target_products_df = pd.concat(
                    [self.target_products_df, scene_target_products_df]
                )

            if self.markers_enabled:
                for actor_name in scene_target_products_df["actor_name"]:
                    actor = self.actors["products"][actor_name]
                    try:
                        target_marker = next(target_markers_iterator[scene_idx])
                    except StopIteration:
                        raise RuntimeError(
                            f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}"
                        )
                    target_marker.set_pose(actor.pose)

    def evaluate(self):
        target_pos = self.calc_target_pose().p
        tolerance = torch.tensor(
            [self.TARGET_POS_THRESH, self.TARGET_POS_THRESH, self.TARGET_POS_THRESH]
        ).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[
                self.target_products_df["scene_idx"] == scene_idx
            ]

            scene_target_products = set(self.TARGET_PRODUCTS_NAMES)
            scene_placed_products = set()

            for _, row in scene_target_products_df.iterrows():
                actor_name = row["actor_name"]
                target_product_pos = self.actors["products"][actor_name].pose.p
                scene_is_target_product_placed = torch.all(
                    (target_product_pos >= (target_pos[scene_idx] - tolerance))
                    & (target_product_pos <= (target_pos[scene_idx] + tolerance)),
                    dim=-1,
                )
                if scene_is_target_product_placed:
                    product_name = row["product_name"]
                    scene_placed_products.add(product_name)
            scene_is_obj_placed = torch.tensor(
                [scene_placed_products == scene_target_products], device=self.device
            )
            is_obj_placed.append(scene_is_obj_placed)

        is_obj_placed = torch.cat(is_obj_placed)

        is_robot_static = self.agent.is_static(0.2)

        is_non_target_products_replaced = torch.zeros_like(is_robot_static, dtype=bool)

        for scene_idx in range(self.num_envs):
            scene_products_df = self.products_df[
                self.products_df["scene_idx"] == scene_idx
            ]

            scene_target_products_df = self.target_products_df[
                self.target_products_df["scene_idx"] == scene_idx
            ]
            non_target_actors = set(scene_products_df["actor_name"]) - set(
                scene_target_products_df["actor_name"]
            )

            for actor_name in non_target_actors:
                actor = self.actors["products"][actor_name]
                if actor_name in self.products_initial_poses:
                    if not torch.all(
                        torch.isclose(
                            actor.pose.raw_pose,
                            self.products_initial_poses[actor_name],
                            rtol=0.1,
                            atol=0.1,
                        )
                    ):
                        is_non_target_products_replaced[scene_idx] = True

                        if self.markers_enabled:
                            # make marker red if non-target product moved
                            render_component = (
                                self.target_volumes[scene_idx][0]
                                ._objs[0]
                                .find_component_by_type(
                                    sapien.pysapien.render.RenderBodyComponent
                                )
                            )
                            render_component.render_shapes[0].material.base_color = [
                                1.0,
                                0.0,
                                0.0,
                                0.5,
                            ]

                        break

        return {
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_non_target_products_displaced": is_non_target_products_replaced,
            "success": is_obj_placed
            & is_robot_static
            & (~is_non_target_products_replaced),
        }

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(self.LANGUAGE_INSTRUCTION)


@register_env("PickToBasketMultiLatteContEnv", max_episode_steps=200000)
class PickToBasketMultiLatteContEnv(PickToBasketMultiContEnv):
    LANGUAGE_INSTRUCTION = "prepare the ingredients for a latte"

    TARGET_PRODUCTS_NAMES = ["auchan milk", "myboo coffee package"]

    # TARGET_PRODUCTS_NAMES = [
    #     ['auchan milk', 'milk', 'plastic milk bottle'],
    #     ['myboo coffee package', 'paper coffee package', 'coffee package', 'capsules dolce gusto', 'nescafe']
    # ]

    # TARGET_PRODUCTS_NAMES = [
    #     {
    #         'candidates': ['auchan milk', 'milk', 'plastic milk bottle'],
    #         'count': 1,
    #         'required': True
    #     },
    #     {
    #         'candidates': ['myboo coffee package', 'paper coffee package', 'coffee package', 'capsules dolce gusto', 'nescafe'],
    #         'count': 1,
    #         'required': True
    #     }
    # ]
