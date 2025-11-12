#!/bin/bash
# wrapper_submit.sh
python scripts/generate_hparams_single_param.py

# --- Submit baseline job ---
BASELINE_JOB_ID=$(sbatch --parsable scripts/test_varying_array_main_single_param.job)
echo "Submitted baseline job with ID: $BASELINE_JOB_ID"

# # --- Submit sweep array with dependency on baseline ---
# SWEEP_JOB_ID=$(sbatch --dependency=afterok:$BASELINE_JOB_ID scripts/test_varying_array_single_param.job)
# echo "Submitted sweep array job with ID: $SWEEP_JOB_ID, dependent on baseline completion"