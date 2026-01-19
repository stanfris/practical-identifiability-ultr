set -u  # NO set -e

MAX_PARALLEL=5

BASELINE_HPARAMS="scripts/hparams_varying_single_experiment_main.txt"
SWEEP_HPARAMS="scripts/hparams_varying_single_experiment.txt"

# -----------------------------
# Generate hparams
# -----------------------------
echo "Generating hparams..."
python scripts/generate_hparams_deterministic_custom_data.py

# -----------------------------
# Portable semaphore runner
# -----------------------------
run_parallel() {
    local hparams_file=$1
    local entrypoint=$2
    local phase_name=$3

    echo "========================================"
    echo "Starting $phase_name"
    echo "========================================"

    while IFS= read -r line || [[ -n "$line" ]]; do

        # Throttle: wait until running jobs < MAX_PARALLEL
        while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
            sleep 0.2
        done

        (
            echo "[START] $line"
            python "$entrypoint" -m $line
            status=$?
            if [[ $status -ne 0 ]]; then
                echo "[FAIL ] exit=$status :: $line" >&2
            else
                echo "[DONE ] $line"
            fi
        ) &

    done < "$hparams_file"

    # Wait for all jobs in this phase
    wait

    echo "========================================"
    echo "$phase_name completed"
    echo "========================================"
}

# -----------------------------
# Baseline phase
# -----------------------------
run_parallel \
    "$BASELINE_HPARAMS" \
    "main.py" \
    "Baseline experiments"

# -----------------------------
# Sweep phase (after baseline)
# -----------------------------
run_parallel \
    "$SWEEP_HPARAMS" \
    "varying.py" \
    "Sweep experiments"