# cuPQC Benchmark System

容器化的 GPU 性能基准测试系统，使用本地 SDK 代码进行编译和测试。

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
4. **本地 cuPQC SDK** (头文件 + 库)

### 目录结构要求

你的本地 SDK 目录应包含:

```
/path/to/your/cupqc-sdk/
├── include/
│   ├── cupqc/           # cuPQC 头文件
│   │   ├── hash.hpp
│   │   └── pk.hpp
│   └── ...
├── lib/
│   ├── libcupqc-hash.a
│   ├── libcupqc-pk.a
│   └── ...
└── ...
```

### 配置

```bash
# 1. 复制配置模板
cd benchmark
cp .env.example .env

# 2. 编辑 .env，设置本地 SDK 路径
nano .env
```

修改 `LOCAL_SDK_PATH` 指向你的 SDK 目录:

```bash
LOCAL_SDK_PATH=/home/username/cupqc-sdk
```

### 构建并运行

```bash
# 一键启动
docker compose up --build
```

### 访问报告

浏览器访问: **http://localhost:8080**

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

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_SDK_PATH` | (必填) | 本地 SDK 目录路径 |
| `BENCHMARK_ITERATIONS` | 10 | 迭代次数 |
| `BENCHMARK_BATCH_SIZES` | 1,10,100,1000,5000 | 批量大小 |
| `CUDA_VISIBLE_DEVICES` | 0 | GPU 编号 |
| `HTTP_PORT` | 8080 | HTTP 端口 |

## 输出文件

```
benchmark/results/
├── index.html           # 可视化报告
├── results_latest.json  # 最新数据
└── results_*.json       # 历史数据
```

## 常见问题

### 1. SDK 路径错误

```
ERROR: No SDK found at /opt/cupqc-sdk
```

确保 `.env` 中的 `LOCAL_SDK_PATH` 正确指向 SDK 目录。

### 2. GPU 不可用

```bash
# 验证 Docker GPU 支持
docker run --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. 编译失败

检查 SDK 目录结构是否正确:
```bash
ls -la /path/to/your/cupqc-sdk/
ls -la /path/to/your/cupqc-sdk/include/cupqc/
ls -la /path/to/your/cupqc-sdk/lib/
```

## 目录结构

```
cuPQC/
├── benchmark/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .env.example
│   ├── scripts/
│   │   ├── run_benchmark.py
│   │   └── web_server.py
│   └── results/
├── examples/
│   ├── hash/
│   └── public_key/
└── (your SDK at LOCAL_SDK_PATH)
```
