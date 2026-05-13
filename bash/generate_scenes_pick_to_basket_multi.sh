#! /bin/bash

WORKERS=2
DATA_PATH='.'

CONFIGS=(
  pick_to_basket_diet
  pick_to_basket_no_alcohol
  pick_to_basket_red_bull
  pick_to_basket_lactose
  pick_to_basket_mojito
  pick_to_basket_british_tea
  pick_to_basket_cuba_libre
  pick_to_basket_grandma_tea
  pick_to_basket_kids_party
  pick_to_basket_overnight_oats
)

for config in "${CONFIGS[@]}"; do
  python scripts/generate_scene_continuous.py \
    ds_continuous=$config \
    ds_continuous.num_workers=$WORKERS \
    assets.assets_dir_path=$DATA_PATH/assets \
    ds_continuous.output_dir=$DATA_PATH/demo_envs/$config
done
