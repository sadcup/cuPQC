# cuPQC Benchmark System

容器化的 GPU 性能基准测试系统，用于测试 NVIDIA cuPQC SDK 的性能。

## 功能特性

- 📦 使用本地 cuPQC SDK 代码
- 🔨 容器内编译示例
- 📊 多批量大小性能测试
- 📈 Chart.js 可视化报告
- 🌐 HTTP 服务访问

## 快速开始

### 前置要求

1. NVIDIA GPU (Compute Capability 7.0+)
2. NVIDIA Driver (525+)
3. Docker + NVIDIA Container Toolkit

### 一键启动

```bash
cd benchmark
docker compose up --build
```

### 访问报告

构建完成后，浏览器访问: **http://localhost:8080**

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BENCHMARK_ITERATIONS` | 10 | 每次测试的迭代次数 |
| `BENCHMARK_BATCH_SIZES` | 1,10,100,1000,5000 | 批量大小列表 |
| `CUDA_VISIBLE_DEVICES` | 0 | 使用的 GPU 编号 |
| `HTTP_PORT` | 8080 | HTTP 服务端口 |

### 自定义参数

创建 `.env` 文件:

```bash
echo "BENCHMARK_ITERATIONS=20" > .env
echo "BENCHMARK_BATCH_SIZES=1,100,1000,5000,10000" >> .env
docker compose up --build
```

## 测试项目

| 类别 | 算法 |
|------|------|
| 哈希函数 | SHA-2 256, SHA-3, Poseidon2, Merkle Tree |
| 公钥密码 | ML-KEM 512, ML-DSA 44 |

## 关键指标

| 指标 | 说明 |
|------|------|
| **吞吐量** | ops/sec - 每秒操作数 |
| **延迟** | 平均执行时间 (ms) |
| **P95** | 95th 百分位延迟 |

## 输出文件

```
benchmark/results/
├── index.html           # 可视化报告
├── results_latest.json # 最新数据
└── results_*.json      # 历史数据
```

## 工作原理

1. **挂载 SDK**: 将 `../cupqc-sdk-0.4.1-x86_64` 挂载到容器内 `/opt/cupqc-sdk`
2. **编译示例**: 使用 nvcc 编译 hash 和 public_key 示例
3. **运行测试**: 对每个算法测试不同批量大小
4. **生成报告**: 创建 HTML 报告 + JSON 数据
5. **启动 HTTP**: 提供 Web 服务访问

## 目录结构

```
cuPQC/
├── benchmark/
│   ├── Dockerfile           # 容器配置
│   ├── docker-compose.yml   # Docker Compose 配置
│   ├── .env.example        # 环境变量模板
│   ├── scripts/
│   │   ├── run_benchmark.py  # Python benchmark 脚本
│   │   ├── run_benchmark.sh  # Shell 启动脚本
│   │   └── web_server.py     # HTTP 服务器
│   └── results/              # 测试结果
│       ├── index.html        # 可视化报告
│       └── results_*.json  # 原始数据
├── examples/
│   ├── hash/
│   └── public_key/
└── cupqc-sdk-0.4.1-x86_64/  # SDK (需要自行下载)
```

## 常见问题

### 1. GPU 不可用

```bash
# 验证 Docker GPU 支持
docker run --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. 编译失败

确保 SDK 目录存在:
```bash
ls -la ../cupqc-sdk-0.4.1-x86_64/
ls -la ../cupqc-sdk-0.4.1-x86_64/include/cupqc/
ls -la ../cupqc-sdk-0.4.1-x86_64/lib/
```

### 3. 端口占用

修改 `.env` 中的 `HTTP_PORT`:

```bash
HTTP_PORT=9000
```

## 示例结果 (RTX 3080 Ti)

| 算法 | Batch=1 延迟 | Batch=5000 吞吐量 |
|------|-------------|-----------------|
| SHA-2 256 | ~212 ms | ~25,000 ops/s |
| SHA-3 | ~200 ms | ~25,000 ops/s |
| Poseidon2 | ~200 ms | ~25,000 ops/s |
| Merkle Tree | ~210 ms | ~24,000 ops/s |
| ML-KEM 512 | ~203 ms | ~24,000 ops/s |
| ML-DSA 44 | ~208 ms | ~24,500 ops/s |

## 停止服务

```bash
# Ctrl+C 停止
# 或
docker compose down
```
