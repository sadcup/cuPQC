# cuPQC SDK Examples

This directory contains examples demonstrating individual cuPQC SDK library functions. These examples showcase GPU-accelerated cryptographic primitives including hash functions, Merkle trees, and post-quantum public-key cryptography algorithms.

## Examples by Library

### Hash Functions (`hash/`)
The hash examples demonstrate various cryptographic hash functions and Merkle tree operations:
- **SHA-2** - SHA-256, a widely-used cryptographic hash function
- **SHA-3** - The latest NIST-standardized hash function
- **Poseidon2** - A zero-knowledge-friendly hash function optimized for ZK circuits
- **Merkle Trees** - Tree construction, proof generation, and verification with performance comparisons

### Public-Key Cryptography (`public_key/`)
The public-key examples demonstrate post-quantum cryptographic algorithms:
- **ML-KEM** - Module-Lattice Key Encapsulation Mechanism (NIST FIPS 203) for secure key exchange
- **ML-DSA** - Module-Lattice Digital Signature Algorithm (NIST FIPS 204) for digital signatures

## Building Examples

Each subdirectory includes a `Makefile`. To build all examples in a directory:

```bash
cd examples/<library>
make
```

The Makefiles expect cuPQC SDK at `/usr/local/cupqc-sdk` or a user-specified path (set `CUPQC_SDK_DIR` environment variable or modify the Makefile).
