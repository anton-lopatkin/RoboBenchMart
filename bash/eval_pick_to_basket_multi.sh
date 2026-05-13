#!/bin/bash

MODEL="${1:-x-ai/grok-4}"
N_EPISODES="${2:-10}"
EXTRA_ARGS="${@:3}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE="${MODEL//\//_}"
LOG_FILE="logs/pick_to_basket_multi_${MODEL_SAFE}_${TIMESTAMP}.log"
RESULTS_FILE="logs/results_pick_to_basket_multi_${MODEL_SAFE}_${TIMESTAMP}.jsonl"
mkdir -p logs

TASKS=(
  "demo_envs/pick_to_basket_diet PickToBasketDietContEnv"
  "demo_envs/pick_to_basket_no_alcohol PickToBasketNoAlcoholContEnv"
  "demo_envs/pick_to_basket_red_bull PickToBasketRedBullContEnv"
  "demo_envs/pick_to_basket_lactose PickToBasketLactoseContEnv"
  "demo_envs/pick_to_basket_mojito PickToBasketMojitoContEnv"
  "demo_envs/pick_to_basket_british_tea PickToBasketBritishTeaContEnv"
  "demo_envs/pick_to_basket_cuba_libre PickToBasketCubaLibreContEnv"
  "demo_envs/pick_to_basket_grandma_tea PickToBasketGrandmaTeaContEnv"
  "demo_envs/pick_to_basket_kids_party PickToBasketKidsPartyContEnv"
  "demo_envs/pick_to_basket_overnight_oats PickToBasketOvernightOatsContEnv"
)

{
  for task in "${TASKS[@]}"; do
    read -r scene_dir env_id <<< "$task"
    python planning/run.py "$scene_dir" -e "$env_id" -m "$MODEL" -n "$N_EPISODES" \
      --results-log "$RESULTS_FILE" $EXTRA_ARGS
  done
} 2>&1 | tee "$LOG_FILE"

echo "Log saved to $LOG_FILE"

if [ -f "$RESULTS_FILE" ]; then
  python - "$RESULTS_FILE" << 'EOF'
import json, sys
from collections import Counter

results_file = sys.argv[1]
rows = []
with open(results_file) as f:
    for line in f:
        d = json.loads(line)
        episodes = d["episodes"]
        n = len(episodes)
        successes = sum(e["success"] for e in episodes)
        crashes = sum(e["crash"] is not None for e in episodes)
        crash_types = Counter(e["crash"] for e in episodes if e["crash"])
        short = d["env_id"].replace("ContEnv", "").replace("PickToBasket", "")
        rows.append((short, successes, n, crashes, crash_types))

total_s = sum(r[1] for r in rows)
total_n = sum(r[2] for r in rows)
total_c = sum(r[3] for r in rows)

model = json.loads(open(results_file).readline())["model"]
print(f"\n{model} evaluation results ")
print(f"{'Task':<32} {'Success':>8}  {'Crashes':>7}  {'Rate':>5}")
print("-" * 58)
for short, s, n, c, _ in rows:
    print(f"{short:<32} {s:>4}/{n:<3}  {c:>7}  {s/n*100:>4.0f}%")
print("-" * 58)
print(f"{'TOTAL':<32} {total_s:>4}/{total_n:<3}  {total_c:>7}  {total_s/total_n*100:>4.0f}%")
EOF
fi
