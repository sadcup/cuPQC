#!/bin/bash
# =============================================================================
# cuPQC Benchmark Runner Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${BENCHMARK_DIR}/results"

# Default values
ITERATIONS=10
BATCH_SIZES="1,10,100,1000,5000"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --batches)
            BATCH_SIZES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "cuPQC Benchmark Runner"
echo "=============================================="
echo "Iterations: $ITERATIONS"
echo "Batch sizes: $BATCH_SIZES"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run Python benchmark
cd "$BENCHMARK_DIR"
python3 scripts/run_benchmark.py \
    --iterations "$ITERATIONS" \
    --batches "$BATCH_SIZES" \
    --output-dir "$RESULTS_DIR"

# Start web server
echo ""
echo "Starting web server on port 8080..."
python3 scripts/web_server.py
