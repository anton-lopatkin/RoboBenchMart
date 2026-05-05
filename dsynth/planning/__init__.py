from dsynth.planning.solve import *
from dsynth.planning.solvers import *

MP_SOLUTIONS = {
    "PickToBasketContNiveaEnv": solve_fetch_pick_to_basket_cont_one_prod_w_skills,
    "PickToBasketContStarsEnv": solve_fetch_pick_to_basket_cont_one_prod_w_skills,
    "PickToBasketContFantaEnv": solve_fetch_pick_to_basket_cont_one_prod_w_skills,

    "MoveFromBoardToBoardVanishContEnv": solve_fetch_move_to_board_cont_one_prod,
    "MoveFromBoardToBoardNestleContEnv": solve_fetch_move_to_board_cont_one_prod,
    "MoveFromBoardToBoardDuffContEnv": solve_fetch_move_to_board_cont_one_prod,

    "PickFromFloorSlamContEnv": solve_fetch_pick_from_floor_cont,
    "PickFromFloorBeansContEnv": solve_fetch_pick_from_floor_cont,

    "OpenDoorShowcaseContEnv": solve_fetch_open_door_showcase_cont,
    "OpenDoorFridgeContEnv": solve_fetch_open_door_fridge_cont,

    "CloseDoorShowcaseContEnv": solve_fetch_close_door_showcase_cont,
    "CloseDoorFridgeContEnv": solve_fetch_close_door_fridge_cont
}