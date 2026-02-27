# AGENTS.md - cuPQC SDK Examples

This file provides guidance for AI agents working in this repository.

## Project Overview

This repository contains examples and applications demonstrating the NVIDIA cuPQC SDK - a GPU-accelerated cryptography library for post-quantum cryptography (ML-KEM, ML-DSA) and hash functions (SHA-2, SHA-3, Poseidon2, Merkle Trees).

- **Language**: CUDA C++ (`.cu` files)
- **Build System**: Makefiles (no CMake)
- **CUDA Version**: 12.8+
- **C++ Standard**: C++17
- **SDK Dependency**: cuPQC 0.4.1+ (must be installed separately)

## Build Commands

### Building Examples

```bash
# Set SDK path (if not at default /usr/local/cupqc-sdk)
export CUPQC_SDK_DIR=/path/to/cupqc-sdk

# Build hash examples
cd examples/hash
make

# Build public key examples
cd examples/public_key
make

# Build applications
cd applications/merkle_proof_poseidon2_bb
make
```

### Building a Single Example

Each Makefile compiles all `.cu` files in the directory. To build a specific example:

```bash
cd examples/hash
make example_sha2    # Builds only example_sha2 from example_sha2.cu
```

### Clean Build

```bash
cd examples/hash
make clean
```

### Running Examples

After building, run the compiled binary:

```bash
./example_sha2
./example_poseidon2
./example_ml_kem
```

## Code Style Guidelines

### File Headers

Every `.cu` file must include the Apache 2.0 license header:

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

### Imports

- Standard library headers first (alphabetically), then third-party, then project
- Use angle brackets for SDK headers: `#include <hash.hpp>`, `#include <pk.hpp>`

```cpp
#include <vector>
#include <iomanip>
#include <iostream>
#include <random>

#include <hash.hpp>
```

### Namespace

Always use the cuPQC namespace:

```cpp
using namespace cupqc;
```

### Naming Conventions

- **Types (classes, structs)**: PascalCase (e.g., `MLKEM512Key`, `SHA2_256_THREAD`)
- **Functions**: camelCase (e.g., `hash_sha2`, `ml_kem_keygen`)
- **Variables**: camelCase (e.g., `public_keys`, `d_msg`)
- **Constants/Enums**: SCREAMING_SNAKE_CASE (e.g., `CUDA_SUCCESS`)
- **CUDA kernels**: snake_case with `_kernel` suffix (e.g., `hash_sha2_kernel`)

### Indentation and Formatting

- **Indentation**: 4 spaces (no tabs)
- **Braces**: K&R style (opening brace on same line)
- **Line length**: Keep lines under 120 characters when practical

### Type Casting

Use C++ casts, not C-style casts:

- `reinterpret_cast<T>` for pointer conversions
- `static_cast<T>` for numeric conversions
- `const_cast<T>` for const removal

```cpp
cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size());
uint8_t val = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
```

### Unused Parameters

Mark unused function parameters with `[[maybe_unused]]`:

```cpp
int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
```

### Error Handling

Check CUDA API return values and use `cudaGetErrorString` for debugging:

```cpp
cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_msg), size);
if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return 1;
}
```

### Memory Management

- Always free CUDA memory with `cudaFree()`
- Use RAII patterns where possible
- Pair allocations with deallocations in the same scope

```cpp
uint8_t* d_msg = nullptr;
cudaMalloc(reinterpret_cast<void**>(&d_msg), size);
// ... use d_msg ...
cudaFree(d_msg);
```

### GPU Kernel Conventions

- Use `__global__` for kernels launched from host
- Use `__device__` for device functions
- Use `__shared__` for shared memory declarations
- Include `cudaDeviceSynchronize()` after kernel launches when needed

### Documentation

Use Doxygen-style comments for functions and complex code blocks:

```cpp
/**
 * Generate Merkle tree from messages
 * msg: Vector of messages (each element is a leaf)
 * inbuf_len: Length of each message
 * merkle: Merkle tree structure
 * d_root: Device pointer to store root hash
 */
void generate_tree(...)
```

## Project Structure

```
cuPQC/
├── applications/
│   └── merkle_proof_poseidon2_bb/    # Full application
├── examples/
│   ├── hash/                         # Hash function examples
│   │   ├── example_sha2.cu
│   │   ├── example_sha3.cu
│   │   ├── example_poseidon2.cu
│   │   └── example_merkle*.cu
│   └── public_key/                   # Public key examples
│       ├── example_ml_kem.cu
│       └── example_ml_dsa.cu
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## SDK Configuration

The code depends on the NVIDIA cuPQC SDK headers:

- **Default path**: `/usr/local/cupqc-sdk`
- **Environment variable**: `CUPQC_SDK_DIR`

Include paths required:
- `-I$(CUPQC_SDK_DIR)/include/cupqc` (cuPQC headers)
- `-I$(CUPQC_SDK_DIR)/include` (CommonDX headers)

Link libraries:
- `-lcupqc-hash` for hash functions
- `-lcupqc-pk` for public key cryptography

## Testing

This repository contains **example applications**, not unit tests. Each example includes self-verification:

- Hash examples compare computed digests against known test vectors
- ML-KEM examples verify encapsulation/decapsulation produce matching shared secrets
- Merkle tree examples verify proof generation and verification

Run examples to verify correct operation - exit code 0 indicates success.

## Common Issues

1. **SDK not found**: Ensure `CUPQC_SDK_DIR` is set or SDK is installed at `/usr/local/cupqc-sdk`
2. **CUDA architecture mismatch**: Use `-arch=native` for local GPU, or specify target architecture (e.g., `-arch=sm_89`)
3. **Linker errors**: Ensure cuPQC libraries are in `LD_LIBRARY_PATH` or linked correctly

## Additional Resources

- [cuPQC Documentation](https://docs.nvidia.com/cuda/cupqc/)
- [cuPQC Developer Hub](https://nvidia.github.io/cuPQC/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
