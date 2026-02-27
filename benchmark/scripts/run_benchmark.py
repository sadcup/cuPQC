#!/usr/bin/env python3
"""
cuPQC Benchmark Runner
=======================
Runs performance benchmarks on cuPQC SDK examples with varying batch sizes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class CUPQCBenchmark:
    def __init__(self, base_dir: str, iterations: int = 10):
        self.base_dir = Path(base_dir)
        self.iterations = iterations
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": self.get_gpu_info(),
            "benchmarks": {}
        }

    def get_gpu_info(self) -> dict:
        """Get GPU information using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "driver": parts[1],
                        "memory_mb": parts[2],
                        "compute_cap": parts[3]
                    })
            return {"gpus": gpus, "count": len(gpus)}
        except Exception as e:
            return {"error": str(e)}

    def run_executable(self, executable: Path, batch_size: int = 1) -> dict:
        """Run an executable and measure performance"""
        if not executable.exists():
            return {"error": f"Executable not found: {executable}"}

        timings = []
        
        for i in range(self.iterations):
            start_time = time.perf_counter()
            
            try:
                result = subprocess.run(
                    [str(executable)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000
                timings.append(elapsed_ms)
                
                if result.returncode != 0:
                    print(f"  [WARN] {executable.name} returned non-zero: {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                print(f"  [ERROR] {executable.name} timed out")
                timings.append(None)
            except Exception as e:
                print(f"  [ERROR] {executable.name} failed: {e}")
                timings.append(None)

        # Filter out None values
        valid_timings = [t for t in timings if t is not None]
        
        if not valid_timings:
            return {"error": "All runs failed", "batch_size": batch_size}
        
        # Calculate statistics
        valid_timings.sort()
        count = len(valid_timings)
        
        return {
            "batch_size": batch_size,
            "iterations": count,
            "min_ms": min(valid_timings),
            "max_ms": max(valid_timings),
            "avg_ms": sum(valid_timings) / count,
            "median_ms": valid_timings[count // 2],
            "p95_ms": valid_timings[int(count * 0.95)] if count > 1 else valid_timings[0],
            "throughput_ops_per_sec": (batch_size * 1000) / (sum(valid_timings) / count),
            "raw_timings_ms": valid_timings
        }

    def run_hash_benchmarks(self, batch_sizes: list):
        """Run hash function benchmarks"""
        print("\n=== Hash Function Benchmarks ===")
        
        examples_dir = self.base_dir / "examples" / "hash"
        
        benchmarks = {
            "sha2": {
                "name": "SHA-2 256",
                "executable": str(examples_dir / "example_sha2")
            },
            "sha3": {
                "name": "SHA-3",
                "executable": str(examples_dir / "example_sha3")
            },
            "poseidon2": {
                "name": "Poseidon2",
                "executable": str(examples_dir / "example_poseidon2")
            },
            "merkle": {
                "name": "Merkle Tree",
                "executable": str(examples_dir / "example_merkle")
            }
        }
        
        for key, info in benchmarks.items():
            print(f"\nBenchmarking {info['name']}...")
            info["results"] = []
            
            for batch in batch_sizes:
                print(f"  Batch size: {batch}")
                result = self.run_executable(Path(info["executable"]), batch)
                info["results"].append(result)
                
            self.results["benchmarks"][key] = info

    def run_pk_benchmarks(self, batch_sizes: list):
        """Run public key cryptography benchmarks"""
        print("\n=== Public Key Cryptography Benchmarks ===")
        
        examples_dir = self.base_dir / "examples" / "public_key"
        
        benchmarks = {
            "ml_kem": {
                "name": "ML-KEM 512",
                "executable": str(examples_dir / "example_ml_kem")
            },
            "ml_dsa": {
                "name": "ML-DSA 44",
                "executable": str(examples_dir / "example_ml_dsa")
            }
        }
        
        for key, info in benchmarks.items():
            print(f"\nBenchmarking {info['name']}...")
            info["results"] = []
            
            for batch in batch_sizes:
                print(f"  Batch size: {batch}")
                result = self.run_executable(Path(info["executable"]), batch)
                info["results"].append(result)
                
            self.results["benchmarks"][key] = info

    def save_results(self, output_path: Path):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, cls=PathEncoder)
        print(f"\nResults saved to: {output_path}")

    def generate_report(self, output_path: Path):
        """Generate HTML report"""
        html = self._generate_html_report()
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"HTML report saved to: {output_path}")

    def _generate_html_report(self) -> str:
        """Generate HTML report with charts"""
        
        benchmarks_json = json.dumps(self.results)
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>cuPQC Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            text-align: center; 
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{ 
            text-align: center; 
            color: #888;
            margin-bottom: 30px;
        }}
        .gpu-info {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        .gpu-info h2 {{ color: #00d4ff; margin-bottom: 15px; }}
        .gpu-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .gpu-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #333;
        }}
        .benchmark-section {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        .benchmark-section h2 {{ 
            color: #00d4ff; 
            margin-bottom: 20px;
            border-bottom: 2px solid #00d4ff;
            padding-bottom: 10px;
        }}
        .chart-container {{
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #00ff88;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 cuPQC Benchmark Report</h1>
        <p class="subtitle">GPU-Accelerated Post-Quantum Cryptography Performance</p>
        
        <div class="gpu-info">
            <h2>🖥️ GPU Information</h2>
            <div class="gpu-grid" id="gpuGrid"></div>
        </div>
        
        <div id="benchmarks"></div>
        
        <p class="timestamp">Generated: {self.results['timestamp']}</p>
    </div>

    <script>
        const benchmarkData = {benchmarks_json};
        
        // Display GPU info
        const gpuGrid = document.getElementById('gpuGrid');
        if (benchmarkData.gpu_info.gpus) {{
            benchmarkData.gpu_info.gpus.forEach(gpu => {{
                gpuGrid.innerHTML += `
                    <div class="gpu-card">
                        <div style="font-weight: bold; color: #00d4ff;">${{gpu.name}}</div>
                        <div style="margin-top: 10px; font-size: 0.9em;">
                            <div>Driver: ${{gpu.driver}}</div>
                            <div>Memory: ${{gpu.memory_mb}} MB</div>
                            <div>Compute Capability: ${{gpu.compute_cap}}</div>
                        </div>
                    </div>
                `;
            }});
        }}
        
        // Color schemes for charts
        const colors = {{
            primary: '#00d4ff',
            secondary: '#00ff88',
            tertiary: '#ff6b6b',
            quaternary: '#ffd93d',
            background: 'rgba(0, 212, 255, 0.1)'
        }};
        
        // Generate benchmark sections
        const benchmarksDiv = document.getElementById('benchmarks');
        
        for (const [key, data] of Object.entries(benchmarkData.benchmarks)) {{
            const section = document.createElement('div');
            section.className = 'benchmark-section';
            
            const results = data.results || [];
            const batchSizes = results.map(r => r.batch_size || 0);
            const avgTimes = results.map(r => r.avg_ms || 0);
            const throughputs = results.map(r => r.throughput_ops_per_sec || 0);
            
            section.innerHTML = `
                <h2>${{data.name}}</h2>
                
                <div class="chart-container">
                    <canvas id="chart-${{key}}-time"></canvas>
                </div>
                
                <div class="chart-container">
                    <canvas id="chart-${{key}}-throughput"></canvas>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${{results.length > 0 ? results[results.length-1].throughput_ops_per_sec.toFixed(0) : 0}}</div>
                        <div class="stat-label">Max Throughput (ops/sec)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{results.length > 0 ? results[results.length-1].avg_ms.toFixed(2) : 0}} ms</div>
                        <div class="stat-label">Avg Latency (batch=${{batchSizes[batchSizes.length-1] || 0}})</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{results.length > 0 ? results[0].min_ms.toFixed(2) : 0}} ms</div>
                        <div class="stat-label">Best Single Run</div>
                    </div>
                </div>
            `;
            
            benchmarksDiv.appendChild(section);
            
            // Create charts after DOM insertion
            setTimeout(() => {{
                // Latency chart
                new Chart(document.getElementById('chart-' + key + '-time'), {{
                    type: 'line',
                    data: {{
                        labels: batchSizes,
                        datasets: [{{
                            label: 'Average Latency (ms)',
                            data: avgTimes,
                            borderColor: colors.primary,
                            backgroundColor: colors.background,
                            fill: true,
                            tension: 0.3
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{ display: true, text: 'Latency vs Batch Size' }},
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{
                                title: {{ display: true, text: 'Batch Size' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            y: {{
                                title: {{ display: true, text: 'Latency (ms)' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
                
                // Throughput chart
                new Chart(document.getElementById('chart-' + key + '-throughput'), {{
                    type: 'line',
                    data: {{
                        labels: batchSizes,
                        datasets: [{{
                            label: 'Throughput (ops/sec)',
                            data: throughputs,
                            borderColor: colors.secondary,
                            backgroundColor: 'rgba(0, 255, 136, 0.1)',
                            fill: true,
                            tension: 0.3
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{ display: true, text: 'Throughput vs Batch Size' }},
                            legend: {{ display: false }}
                        }},
                        scales: {{
                            x: {{
                                title: {{ display: true, text: 'Batch Size' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }},
                            y: {{
                                title: {{ display: true, text: 'Operations/sec' }},
                                grid: {{ color: 'rgba(255,255,255,0.1)' }}
                            }}
                        }}
                    }}
                }});
            }}, 100);
        }}
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="cuPQC Benchmark Runner")
    parser.add_argument(
        "--base-dir", 
        default="/workspace",
        help="Base directory containing examples"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=10,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--batches",
        type=str,
        default="1,10,100,1000",
        help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/benchmark/results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    batch_sizes = [int(b.strip()) for b in args.batches.split(",")]
    
    print("=" * 60)
    print("cuPQC Benchmark Runner")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Iterations: {args.iterations}")
    print(f"Batch sizes: {batch_sizes}")
    
    # Create benchmark instance
    benchmark = CUPQCBenchmark(args.base_dir, args.iterations)
    
    # Display GPU info
    print("\n=== GPU Information ===")
    if benchmark.results["gpu_info"]["gpus"]:
        for gpu in benchmark.results["gpu_info"]["gpus"]:
            print(f"  {gpu['name']} (Compute {gpu['compute_cap']}, {gpu['memory_mb']} MB)")
    else:
        print("  Warning: No GPU detected")
    
    # Run benchmarks
    benchmark.run_hash_benchmarks(batch_sizes)
    benchmark.run_pk_benchmarks(batch_sizes)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark.save_results(output_dir / f"results_{timestamp}.json")
    benchmark.save_results(output_dir / "results_latest.json")
    
    # Generate HTML report
    benchmark.generate_report(output_dir / "index.html")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print(f"\nTo view the report, open: http://localhost:8080/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
