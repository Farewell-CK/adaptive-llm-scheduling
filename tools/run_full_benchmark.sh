#!/bin/bash
set -e

# ==================================================================================
# AdaSplit Full Benchmark Suite Runner
# ==================================================================================
# This script runs the complete set of experiments:
# 1. Monolithic Baseline (Experiment 1)
# 2. Static Partitioning (Experiment 2)
# 3. AdaSplit Dynamic (Experiment 3)
#
# It iterates over a list of QPS and Durations.
# ==================================================================================

# 1. Setup Environment
# Ensure we use the 'vllm' conda environment.
export CONDA_BASE="/data/miniconda"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate vllm
else
    echo "❌ Error: Conda not found at $CONDA_BASE"
    exit 1
fi

# Verify Python
echo "🔍 Environment Check:"
which python3
python3 -c "import vllm; print(f'   vLLM Version: {vllm.__version__}')" || exit 1

# 2. Configuration
# You can override these variables.
QPS_LIST="${QPS_LIST:-1.0,2.0,3.0,4.0}"
DURATION="${DURATION:-30}" # Minutes per experiment run

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SUITE="$SCRIPT_DIR/run_suite.py"

echo "========================================================"
echo "🚀 Starting Full Benchmark Campaign"
echo "   QPS List: $QPS_LIST"
echo "   Duration: $DURATION min"
echo "========================================================"

# Function to run an experiment suite
run_exp() {
    local exp_id=$1
    local name=$2
    echo ""
    echo "--------------------------------------------------------"
    echo "▶️  Starting Experiment $exp_id: $name"
    echo "--------------------------------------------------------"
    
    # Run the suite
    # run_suite.py handles the inner loop of QPS/Duration
    python3 "$RUN_SUITE" --exp "$exp_id" --qps "$QPS_LIST" --min "$DURATION"
    
    echo "✅ Experiment $exp_id Finished."
    
    # Cool down
    echo "❄️  Cooling down for 30s..."
    sleep 30
}

# 3. Execution
run_exp 1 "Monolithic Baseline"
run_exp 2 "Static Partitioning"
run_exp 3 "AdaSplit Dynamic"

echo ""
echo "========================================================"
echo "🎉 All Experiments Completed Successfully!"
echo "========================================================"
