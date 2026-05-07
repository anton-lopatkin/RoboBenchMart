import numpy as np
np.set_printoptions(suppress=True)


from dsynth.envs import *

from dsynth.planning.motionplanner import (
    FetchMotionPlanningSapienSolver
)
from dsynth.planning.utils import (
    get_base_pose,
    BAD_ENV_ERROR_CODE
)
from dsynth.planning.fetch_skills import *


def solve_fetch_pick_to_basket_cont_one_prod_w_skills(env: PickToBasketContEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed, options={'reconfigure': True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        verbose=True,
    )

    if len(planner.planner.planning_world.check_collision()) > 0:
        return BAD_ENV_ERROR_CODE
        
    env = env.unwrapped
    # -------------------------------------------------------------------------- #
    # Setup target product
    # -------------------------------------------------------------------------- #

    #find the closest to gripper as target product
    max_dist = np.inf
    target_product_name = ''
    for target_actor_name in env.target_products_df['actor_name']:
        prod_pos = env.actors['products'][target_actor_name].pose.sp.p
    
        if np.linalg.norm(prod_pos - get_base_pose(env).p) < max_dist:
            max_dist = np.linalg.norm(prod_pos - get_base_pose(env).p)
            target_product_name = target_actor_name

    target_product_actor = env.actors['products'][target_product_name]

    # -------------------------------------------------------------------------- #
    # Align to target product
    # -------------------------------------------------------------------------- #

    res = align_to_target_product(env, planner, target_product_actor)
    if res == -1:
        return res
    
    # -------------------------------------------------------------------------- #
    # Fetch object from shelf
    # -------------------------------------------------------------------------- #

    res = fetch_object_from_shelf(env, planner, target_product_actor, n_grasps=10, num_tries=5)
    if res == -1:
        return res

    return drop_to_basket(env, planner)
