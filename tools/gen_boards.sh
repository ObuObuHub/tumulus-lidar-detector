#!/bin/zsh
# gen_boards.sh — generează planșe de validat din mai multe zone Oltenia PROASPETE (secvențial, memory-safe).
# Per zonă: sweep (test_recall_area, descarcă+scorează) -> copie detecții la nume de zonă -> board scor-ascuns.
# ⚠ test_recall_area scrie HARDCODAT /tmp/catane_detections.csv -> copiez imediat. NU rula 2 sweep-uri în paralel.
cd ~/lidar-match || exit 1
PY=./venv/bin/python
ZONES=(
  "seaca 23.203 43.934 6"
  "giubega 23.438 44.133 6"
  "bistret 23.461 43.899 6"
  "golenti 23.055 43.988 6"
  "calafat 22.992 43.977 6"
)
for row in "${ZONES[@]}"; do
  set -- ${=row}; name=$1; lon=$2; lat=$3; km=$4
  echo "=== SWEEP $name ($lon $lat $km km) $(date +%H:%M:%S) ==="
  $PY -u tools/test_recall_area.py $lon $lat $km > /tmp/sweep_$name.log 2>&1
  if [[ ! -f /tmp/catane_detections.csv ]]; then echo "  !! $name sweep FAIL (fără detecții)"; continue; fi
  cp /tmp/catane_detections.csv /tmp/${name}_detections.csv
  nd=$(( $(wc -l < /tmp/${name}_detections.csv) - 1 ))
  echo "=== BOARD $name ($nd detecții) ==="
  $PY tools/board_from_detections.py /tmp/${name}_detections.csv $HOME/lidar-match/review/board_$name.png /tmp/${name}_map.csv "$name" >> /tmp/sweep_$name.log 2>&1
  echo "  -> review/board_$name.png ; map /tmp/${name}_map.csv"
done
echo "=== TOATE PLANSELE GATA $(date +%H:%M:%S) ==="
ls -la $HOME/lidar-match/review/board_*.png 2>/dev/null
