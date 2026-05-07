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

def solve_fetch_move_to_board_cont_one_prod_w_skills(env: PickToBasketContEnv, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(target_product_actor)
    target_center = np.array(obb.primitive.transform)[:3, 3]
    target_center[2] += env.get_interboard_height() + 0.05
    direction = env.directions_to_shelf[0]
    closing = np.cross(direction, [0., 0., 1.])
    final_pose = env.agent.build_grasp_pose(direction, closing, target_center)

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

    # -------------------------------------------------------------------------- #
    # Align to final pose
    # -------------------------------------------------------------------------- #

    res = align_to_target_pose(env, planner, final_pose)
    if res == -1:
        planner.render_wait()
        return res

    # -------------------------------------------------------------------------- #
    # Place object to position
    # -------------------------------------------------------------------------- #

    res = place_object_to_pos(env, planner, target_center, direction, n_grasps=10)
    if res == -1:
        return res

    res = planner.idle_steps(t=1)
    return res
