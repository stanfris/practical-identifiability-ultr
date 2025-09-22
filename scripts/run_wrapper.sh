#!/bin/bash
# wrapper_submit.sh

# --- Submit baseline job ---
BASELINE_JOB_ID=$(sbatch --parsable scripts/test_varying_array_main.job)
echo "Submitted baseline job with ID: $BASELINE_JOB_ID"

# --- Submit sweep array with dependency on baseline ---
SWEEP_JOB_ID=$(sbatch --dependency=afterok:$BASELINE_JOB_ID scripts/test_varying_array.job)
echo "Submitted sweep array job with ID: $SWEEP_JOB_ID, dependent on baseline completion"