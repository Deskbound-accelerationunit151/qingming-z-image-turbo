# qingming-z-image-turbo
AMD Radeon RX 7900 XTX 24GB（gfx1100）的 Z-Image-Turbo 原生 HIP/C++ 推理实现。  
Native HIP/C++ inference for Z-Image-Turbo, deeply optimized for AMD Radeon RX 7900 XTX 24GB (gfx1100).

## 1. 项目说明 / Overview

感谢 Z-Image 官方团队的杰出工作。  
Special thanks to the Z-Image team for their outstanding work.

- 官方论文 / Paper: [Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer](https://arxiv.org/abs/2511.22699)
- 官方代码 / Official repository: [Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- 官方模型 / Official model: [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [ModelScope](https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo)

> 本项目是非官方社区实现，与 Tongyi-MAI、Alibaba Group、AMD 或其关联方不存在隶属、赞助或官方合作关系。  
> This is an unofficial community implementation. It is not affiliated with, sponsored by, or endorsed by Tongyi-MAI, Alibaba Group, AMD, or their affiliates.

本版本面向RX 7900 XTX的底层执行优化：自定义 gfx1100 BF16 WMMA、持久化 GEMM、融合 QKV/Attention/FFN、静态分辨率数据流、全 GPU 常驻图、单次图启动、无中间 CPU 回传。  
This project is a device-level implementation for RX 7900 XTX: custom gfx1100 BF16 WMMA, persistent GEMM, fused QKV/Attention/FFN, static-resolution dataflow, a fully GPU-resident graph, one graph launch, and no intermediate CPU transfers.

推理不依赖 PyTorch、Diffusers、ComfyUI 或 Python 运行时；编译和运行仅使用原生 C++/HIP/ROCm。 FFmpeg 仅用于图片格式转换。  
Inference requires no PyTorch, Diffusers, ComfyUI, or Python runtime. The binaries use native C++/HIP/ROCm; FFmpeg is only used for image conversion.

## 2. 效果与性能 / Results

### 2.1 图片对比 / Image Comparison

PPM 转 JPG 示例 / PPM to JPG example:

```bash
ffmpeg -y -i outputs/bf16_512x512.ppm -q:v 2 outputs/bf16_512x512.jpg
```

| 模型 / Model | 分辨率 / Resolution | 图片 / Image                                                                |
| ---------- | ---------------: | ------------------------------------------------------------------------- |
| BF16       |          512x512 | <img src="outputs/bf16_512x512.jpg" alt="BF16 512x512" width="240">       |
| BF16       |         576x1024 | <img src="outputs/bf16_576x1024.jpg" alt="BF16 576x1024" width="180">     |
| BF16       |        1024x1024 | <img src="outputs/bf16_1024x1024.jpg" alt="BF16 1024x1024" width="240">   |
| Q8_0       |          512x512 | <img src="outputs/q8_512x512.jpg" alt="Q8_0 512x512" width="240">         |
| Q8_0       |         576x1024 | <img src="outputs/q8_576x1024.jpg" alt="Q8_0 576x1024" width="180">       |
| Q8_0       |        1024x1024 | <img src="outputs/q8_1024x1024.jpg" alt="Q8_0 1024x1024" width="240">     |
| Q6_K       |          512x512 | <img src="outputs/q6k_512x512.jpg" alt="Q6_K 512x512" width="240">        |
| Q6_K       |         576x1024 | <img src="outputs/q6k_576x1024.jpg" alt="Q6_K 576x1024" width="180">      |
| Q6_K       |        1024x1024 | <img src="outputs/q6k_1024x1024.jpg" alt="Q6_K 1024x1024" width="240">    |
| Q5_K_M     |          512x512 | <img src="outputs/q5km_512x512.jpg" alt="Q5_K_M 512x512" width="240">     |
| Q5_K_M     |         576x1024 | <img src="outputs/q5km_576x1024.jpg" alt="Q5_K_M 576x1024" width="180">   |
| Q5_K_M     |        1024x1024 | <img src="outputs/q5km_1024x1024.jpg" alt="Q5_K_M 1024x1024" width="240"> |

### 2.2 性能矩阵 / Performance Matrix

Peak VRAM = Free VRAM before - Free VRAM after workspace.

| 模型 / Model | 分辨率 / Resolution | Full Graph (ms) | DiT (ms) | VAE (ms) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| BF16 | 512x512 | 10225.291 | 7796.849 | 333.667 | 22260 |
| BF16 | 576x1024 | 17718.767 | 14440.581 | 761.190 | 22452 |
| BF16 | 1024x1024 | 31492.927 | 27593.906 | 1602.662 | 22740 |
| Q8_0 | 512x512 | 7406.978 | 7020.802 | 307.687 | 16846 |
| Q8_0 | 576x1024 | 14911.177 | 14064.538 | 767.498 | 17038 |
| Q8_0 | 1024x1024 | 28321.810 | 26629.473 | 1613.359 | 17342 |
| Q6_K | 512x512 | 5596.383 | 5206.913 | 313.494 | 15670 |
| Q6_K | 576x1024 | 11906.525 | 11055.292 | 774.023 | 15862 |
| Q6_K | 1024x1024 | 23273.056 | 21581.111 | 1615.755 | 16166 |
| Q5_K_M | 512x512 | 5675.927 | 5286.444 | 312.845 | 15940 |
| Q5_K_M | 576x1024 | 11912.818 | 11063.576 | 774.640 | 16132 |
| Q5_K_M | 1024x1024 | 23463.243 | 21770.791 | 1618.238 | 16436 |

### 2.3 测试环境 / Test Environment

| 项目 / Item | 配置 / Configuration |
|---|---|
| OS | Ubuntu 24.04 |
| ROCm | 7.2.4 |
| GPU | AMD Radeon RX 7900 XTX 24GB (gfx1100) |
| Image tool | FFmpeg |

## 3. 模型下载 / Model Download

```bash
chmod +x scripts/download.sh
bash scripts/download.sh all
```

按需下载 / Download individually:

```bash
bash scripts/download.sh base
bash scripts/download.sh q8
bash scripts/download.sh q6k
bash scripts/download.sh q5km
```

模型目录 / Model directories:

```text
models/z_image_turbo_ms/
models/z_image_turbo_gguf/z_image_turbo-Q8_0.gguf
models/z_image_turbo_gguf/z_image_turbo-Q6_K.gguf
models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf
```

## 4. 编译 / Build

```bash
COMMON_FLAGS=(
  --offload-arch=gfx1100
  -x hip
  -std=c++17
  -O3
  -ffast-math
  -DNDEBUG
  -Wno-unused-function
  -Wno-unused-result
)

hipcc "${COMMON_FLAGS[@]}" devices/rx7900xtx-24g/bf16.cpp -o zimage_bf16
hipcc "${COMMON_FLAGS[@]}" devices/rx7900xtx-24g/q8.cpp   -o zimage_q8
hipcc "${COMMON_FLAGS[@]}" devices/rx7900xtx-24g/q6k.cpp  -o zimage_q6k
hipcc "${COMMON_FLAGS[@]}" devices/rx7900xtx-24g/q5km.cpp -o zimage_q5km
```

## 5. 运行 / Run

### 5.1 模式 1：单次生成 / Mode 1: One-shot Generation

#### 5.1.1 BF16

##### a. 512x512

```bash
./zimage_bf16 1 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/bf16_512x512.ppm
```

##### b. 576x1024

```bash
./zimage_bf16 1 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/bf16_576x1024.ppm
```

##### c. 1024x1024

```bash
./zimage_bf16 1 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/bf16_1024x1024.ppm
```

#### 5.1.2 Q8_0

##### a. 512x512

```bash
./zimage_q8 1 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q8_0.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q8_512x512.ppm
```

##### b. 576x1024

```bash
./zimage_q8 1 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q8_0.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q8_576x1024.ppm
```

##### c. 1024x1024

```bash
./zimage_q8 1 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q8_0.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q8_1024x1024.ppm
```

#### 5.1.3 Q6_K

##### a. 512x512

```bash
./zimage_q6k 1 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q6_K.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q6k_512x512.ppm
```

##### b. 576x1024

```bash
./zimage_q6k 1 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q6_K.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q6k_576x1024.ppm
```

##### c. 1024x1024

```bash
./zimage_q6k 1 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q6_K.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q6k_1024x1024.ppm
```

#### 5.1.4 Q5_K_M

##### a. 512x512

```bash
./zimage_q5km 1 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q5km_512x512.ppm
```

##### b. 576x1024

```bash
./zimage_q5km 1 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q5km_576x1024.ppm
```

##### c. 1024x1024

```bash
./zimage_q5km 1 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf \
  --prompt "A cinematic portrait in the rain" \
  --seed 42 \
  --max-steps 8 \
  --output outputs/q5km_1024x1024.ppm
```

### 5.2 模式 2：模型常驻 / Mode 2: Resident Interactive Mode

进入后可直接输入提示词；使用 `/help` 查看命令，使用 `/quit` 退出。  
Enter prompts directly; use `/help` for commands and `/quit` to exit.

#### 5.2.1 BF16

##### a. 512x512

```bash
./zimage_bf16 2 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --max-steps 8 \
  --output-dir outputs/bf16_512x512
```

##### b. 576x1024

```bash
./zimage_bf16 2 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --max-steps 8 \
  --output-dir outputs/bf16_576x1024
```

##### c. 1024x1024

```bash
./zimage_bf16 2 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --max-steps 8 \
  --output-dir outputs/bf16_1024x1024
```

#### 5.2.2 Q8_0

##### a. 512x512

```bash
./zimage_q8 2 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q8_0.gguf \
  --max-steps 8 \
  --output-dir outputs/q8_512x512
```

##### b. 576x1024

```bash
./zimage_q8 2 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q8_0.gguf \
  --max-steps 8 \
  --output-dir outputs/q8_576x1024
```

##### c. 1024x1024

```bash
./zimage_q8 2 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q8_0.gguf \
  --max-steps 8 \
  --output-dir outputs/q8_1024x1024
```

#### 5.2.3 Q6_K

##### a. 512x512

```bash
./zimage_q6k 2 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q6_K.gguf \
  --max-steps 8 \
  --output-dir outputs/q6k_512x512
```

##### b. 576x1024

```bash
./zimage_q6k 2 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q6_K.gguf \
  --max-steps 8 \
  --output-dir outputs/q6k_576x1024
```

##### c. 1024x1024

```bash
./zimage_q6k 2 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q6_K.gguf \
  --max-steps 8 \
  --output-dir outputs/q6k_1024x1024
```

#### 5.2.4 Q5_K_M

##### a. 512x512

```bash
./zimage_q5km 2 \
  --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf \
  --max-steps 8 \
  --output-dir outputs/q5km_512x512
```

##### b. 576x1024

```bash
./zimage_q5km 2 \
  --size 576x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf \
  --max-steps 8 \
  --output-dir outputs/q5km_576x1024
```

##### c. 1024x1024

```bash
./zimage_q5km 2 \
  --size 1024x1024 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf models/z_image_turbo_gguf/z_image_turbo-Q5_K_M.gguf \
  --max-steps 8 \
  --output-dir outputs/q5km_1024x1024
```

## 6. 协议 / License

Copyright 2026 qingming-z-image-turbo contributors.

除第三方组件、上游代码及模型资产外，本项目原创源代码和文档依据 **Apache License 2.0** 发布。  
Except for third-party components, upstream code, and model assets, the original source code and documentation in this project are licensed under the **Apache License 2.0**.

- 许可证全文 / Full license: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- 归属声明 / Attribution notices: [`NOTICE`](NOTICE)
- 使用、修改和再分发时，必须遵守 Apache License 2.0，包括保留适用的版权、专利、商标和归属声明，并对修改过的文件作出明确标记。  
  Use, modification, and redistribution must comply with Apache License 2.0, including retention of applicable copyright, patent, trademark, and attribution notices, and prominent identification of modified files.
- Apache License 2.0 不授予 Tongyi-MAI、Alibaba、AMD 或其他权利人的商标使用许可，合理描述作品来源的情况除外。  
  Apache License 2.0 does not grant trademark rights in the names of Tongyi-MAI, Alibaba, AMD, or other rights holders, except for reasonable use required to describe the origin of the work.

### 第三方项目与模型 / Third-Party Projects and Models

- **Z-Image / Z-Image-Turbo** 由 Tongyi-MAI 发布；官方代码仓库和官方模型页面标注为 Apache License 2.0。  
  **Z-Image / Z-Image-Turbo** is published by Tongyi-MAI; its official source repository and official model page identify the project and model as Apache License 2.0.
- 本仓库不声明对 Z-Image、Z-Image-Turbo、其论文、模型权重、训练数据或相关商标的所有权。  
  This repository claims no ownership of Z-Image, Z-Image-Turbo, their paper, model weights, training data, or related trademarks.
- `scripts/download.sh` 下载的模型文件不属于本项目原创代码。模型权重、GGUF 量化文件及其他第三方文件仍受其来源仓库、发布者和适用许可证约束。  
  Model files downloaded by `scripts/download.sh` are not original project code. Model weights, GGUF quantizations, and other third-party files remain governed by their source repositories, distributors, and applicable licenses.
- 再分发模型或量化权重前，使用者必须自行确认并遵守相应的许可证、归属、出口管制及使用限制。  
  Before redistributing models or quantized weights, users are responsible for verifying and complying with all applicable licenses, attribution requirements, export controls, and use restrictions.

### 免责声明 / Disclaimer

本软件按“原样”提供，不附带任何明示或默示保证，包括但不限于适销性、特定用途适用性和不侵权保证。使用者自行承担编译、运行、模型下载、生成内容及再分发所产生的风险和责任。  
This software is provided **“AS IS”**, without warranties or conditions of any kind, express or implied, including merchantability, fitness for a particular purpose, and non-infringement. Users assume all risks and responsibilities arising from compilation, execution, model downloads, generated content, and redistribution.
