# =============================================================================
# cuPQC Benchmark Container
# =============================================================================
# Base: Use existing PyTorch image with CUDA (faster than pulling new image)
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.01-py3

LABEL maintainer="cuPQC Benchmark"
LABEL description="GPU-accelerated Post-Quantum Cryptography Benchmark"

# -----------------------------------------------------------------------------
# Create working directories
# -----------------------------------------------------------------------------
WORKDIR /workspace
RUN mkdir -p /workspace/benchmark/results /opt/cupqc-sdk

# -----------------------------------------------------------------------------
# Copy project files (examples)
# -----------------------------------------------------------------------------
COPY examples/ /workspace/examples/
COPY benchmark/scripts/ /workspace/benchmark/scripts/

# -----------------------------------------------------------------------------
# Expose HTTP port for report
# -----------------------------------------------------------------------------
EXPOSE 8080

# Default command - will be overridden by docker-compose
CMD ["/bin/bash", "-c", "echo 'Use docker-compose to run benchmark'"]
