#!/bin/bash

# Sequentially run through all hyperparameter lines in the file

HPARAMS_FILE="scripts/hparams_varying_single_experiment_main.txt"


# Loop over each line in the file
LINENUM=0
while IFS= read -r LINE; do
    LINENUM=$((LINENUM + 1))
    echo ">>> Running job $LINENUM with params: $LINE"
    python main.py -m $LINE
    echo ">>> Finished job $LINENUM"
    echo "----------------------------------------"
done < "$HPARAMS_FILE"


# HPARAMS_FILE="scripts/hparams_varying_single_experiment.txt"

# # Loop over each line in the file
# LINENUM=0
# while IFS= read -r LINE; do
#     LINENUM=$((LINENUM + 1))
#     echo ">>> Running job $LINENUM with params: $LINE"
#     python varying.py -m $LINE
#     echo ">>> Finished job $LINENUM"
#     echo "----------------------------------------"
# done < "$HPARAMS_FILE"