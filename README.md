# cuPQC SDK Applications

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20x86__64%20%7C%20ARM64-lightgrey)](https://developer.nvidia.com/cupqc)

This repository contains examples demonstrating the capabilities of NVIDIA cuPQC SDK.

**What's included:**
- Example applications with build configs
- Reference implementations
- **GPU Benchmark System** for performance testing


## About cuPQC SDK
NVIDIA cuPQC is a GPU-accelerated cryptography SDK containing specialized libraries for building high-performance cryptographic applications:

- **cuPQC-HASH** - Hash functions & Merkle Trees
- **cuPQC-PK** - Public-Key Cryptography (ML-KEM & ML-DSA)


## Applications & Examples

**Applications:** Browse [applications/](applications/) directory for application implementations.

**Examples:** Browse [examples/](examples/) directory for examples demonstrating individual SDK primitives:
- **Hash Functions** (SHA-2, SHA-3, Poseidon2, Merkle Trees)
- **Public-Key Cryptography** (ML-KEM, ML-DSA)

**Benchmark:** See [benchmark/](benchmark/) directory for automated GPU performance testing.


## Quick Start

### Prerequisites

| Requirement | Specification |
|:------------|:--------------|
| **GPU** | Compute Capability 7.0+ (SM 70, 75, 80, 86, 87, 89, 90) |
| **CUDA** | 12.8 or newer |
| **Compiler** | C++17 (GCC 7+, Clang 9+) |
| **Docker** | With NVIDIA Container Toolkit |
| **SDK** | cuPQC 0.4.1+ (included in benchmark) |

### Download cuPQC SDK (for local development)
[Download the SDK](https://developer.download.nvidia.com/cupqc-download/)

Or use the automated benchmark which includes SDK setup.

### Build Examples

```bash
# Set SDK path (if not at default /usr/local/cupqc-sdk)
export CUPQC_SDK_DIR=/path/to/cupqc-sdk

# Build hash examples
cd examples/hash
make

# Build public key examples
cd examples/public_key
make
```

### Run Examples

```bash
cd examples/hash
./example_sha2
./example_sha3
./example_poseidon2
./example_merkle

cd examples/public_key
./example_ml_kem
./example_ml_dsa
```

---

## GPU Benchmark

Automated performance testing system for cuPQC algorithms. Uses Docker with GPU passthrough.

### Features

- 🚀 Automated build & benchmark execution
- 📊 Multi-batch-size performance testing
- 📈 Chart.js visualization (latency & throughput)
- 🌐 HTTP web server for report access

### Quick Start

```bash
cd benchmark

# Build and run (one command)
docker compose up --build
```

### Access Report

Open browser: **http://localhost:8080**

### Benchmark Parameters

Customize in `.env` file:

```bash
# Example .env
BENCHMARK_ITERATIONS=10
BENCHMARK_BATCH_SIZES=1,10,100,1000,5000
HTTP_PORT=8080
```

### Tested Algorithms

| Category | Algorithms |
|----------|------------|
| Hash Functions | SHA-2 256, SHA-3, Poseidon2, Merkle Tree |
| Public Key | ML-KEM 512, ML-DSA 44 |

### Output Files

- `results/index.html` - Visual report with charts
- `results/results_latest.json` - Raw benchmark data
- `results/results_*.json` - Historical data

For detailed benchmark documentation, see [benchmark/README.md](benchmark/README.md).


## Documentation & Resources
**Official Resources**
- [Product Page](https://developer.nvidia.com/cupqc) - Overview, features, and performance
- [Documentation](https://docs.nvidia.com/cuda/cupqc/) - Complete guides and API reference
- [Download Page](https://developer.nvidia.com/cupqc-download/) - Get the latest release
- [Developer Hub](https://nvidia.github.io/cuPQC/) - Blogs, examples, and application guides

**Community**
- [Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/quantum-computing/cupqc/703) - Community support
- [GitHub Issues](https://github.com/NVIDIA/cuPQC/issues) - Report bugs and request features

**Citation**
If you use cuPQC SDK in a publication, please cite it. Click the "Cite this repository" button in the About section above, or see [CITATION.cff](CITATION.cff) for details.


## License
This project is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.

**Note:** These samples require the NVIDIA cuPQC SDK. By downloading and using the cuPQC SDK, you agree to fully comply with the terms and conditions of the [NVIDIA Software License Agreement](https://docs.nvidia.com/cuda/cupqc/additional/license.html).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.
