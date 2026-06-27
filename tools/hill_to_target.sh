#!/bin/bash
# hill_to_target.sh [TARGET=10000] — apendeaza negative de DEAL pana atinge TARGET in dataset_neg_ro_hill/.
# Loturi succesive cu seed nou; se opreste la stagnare (teren deal epuizat in acoperire). UN SINGUR proces
# (nu rula doua generatoare pe acelasi folder simultan — cursa pe index).
cd ~/lidar-match || exit 1
TARGET=${1:-10000}
seed=100
while true; do
  n=$(ls dataset_neg_ro_hill/*.png 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge "$TARGET" ]; then echo "ATINS $n >= $TARGET"; break; fi
  need=$((TARGET - n)); seed=$((seed + 13))
  echo "=== runda seed=$seed: am $n, mai trebuie $need ==="
  ./venv/bin/python tools/ro_negatives_plain.py "$need" 1500 "$seed" 20 90 hill
  n2=$(ls dataset_neg_ro_hill/*.png 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n2" -le "$n" ]; then echo "STAGNARE ($n2 <= $n) — teren deal epuizat in acoperire, opresc"; break; fi
done
echo "FINAL_HILL=$(ls dataset_neg_ro_hill/*.png 2>/dev/null | wc -l)"
