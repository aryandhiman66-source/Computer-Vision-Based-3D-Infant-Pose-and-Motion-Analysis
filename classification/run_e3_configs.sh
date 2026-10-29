#!/usr/bin/env bash
# E3: architecture-diverse ensemble over the paper's other VID configurations (Supp. Table 2),
# each with mirror/rotation augmentation, 4 runs, seed 0. Waits for E1/E2 to finish first.
cd "$(dirname "$0")/.."
PY=/c/Users/aryan/miniconda3/python.exe
until grep -q "fold 9: paper-selected\|Traceback" classification/runs_aug8_rel_r8.log 2>/dev/null; do sleep 60; done
#      id  kn        kw        fc
for cfg in "2 4,32,64 13,17,9 256" "3 8,8,64 25,13,33 128" "5 4,32,16 25,25,13 256" \
           "6 4,8,64 25,25,25 256" "7 4,8,64 13,7,13 128" "9 8,32,64 13,7,9 256" "10 4,16,64 13,25,13 128"; do
  set -- $cfg
  $PY -u classification/faithful.py --runs 4 --aug 8 --kn $2 --kw $3 --fc $4 --tag aug8_cfg$1_r4 \
      > classification/runs_aug8_cfg$1_r4.log 2>&1
done
