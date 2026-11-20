# bge-small-zh → Core ML：把 ONNX Runtime 从 iOS 包里摘出去

为一个 23MB 的 BERT 模型，iOS 包里实际多背了 **30–40MB 的 ONNX Runtime 静态库**。
本仓库把推理后端换成 **Core ML**（系统框架、零链接成本），模型体积不变，整包 **−40MB**，检索精度反而更好。

| 指标 | ORT int8 | Core ML int8 |
| :--- | :--- | :--- |
| 模型体积 | 23MB `.ort` | 23MB `.mlmodelc` |
| 运行时体积 | 30–40MB（静态库进主二进制） | **0（系统框架）** |
| vs fp32 余弦 | 0.9704 | **0.9995** |
| Top-5 检索重合 | 4.83 / 5 | **5.00 / 5** |
| 单条延迟（Mac，512 token） | 1.1ms | 4.3ms |

另外，2026-11-17 起 `github.com/microsoft/onnxruntime` 及关联仓库（含 swift 包）曾整体 404，
SwiftPM 直接依赖它的项目**构建会断**——这也是迁移的一个现实理由。

---

## 快速开始

```bash
# 0) 环境（建议隔离 venv）
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) 模型（HuggingFace 目录，需含 config.json / vocab.txt / 权重）
#    可用: huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./bge-small-zh
#    国内可用 HF_ENDPOINT=https://hf-mirror.com

# 2) 转换 + int8 量化（产出 .mlpackage）
python convert.py --model-dir ./bge-small-zh \
                  --out ./out/bge_int8.mlpackage \
                  --quant int8 --pooling cls --deployment iOS16

# 3) 离线编译成 .mlmodelc（不需要 Xcode 工程）
xcrun coremlcompiler compile ./out/bge_int8.mlpackage ./out
# → ./out/bge_int8.mlmodelc（23MB），拷进 App target 即可

# 4) 数值校验：Core ML vs PyTorch fp32 真值（可选再加 --ort 对比旧实现）
python validate.py --cm ./out/bge_int8.mlpackage \
                   --model-dir ./bge-small-zh --vocab ./bge-small-zh/vocab.txt
```

期望输出（12 条真实风格中文文本）：

```
CoreML vs fp32           mean=0.9995  min=0.9991  max=0.9999
CoreML Top-5 重合        mean=5.00 / 5
Core ML : 4.3 ms
PyTorch : 4.4 ms
```

---

## 文件说明

| 文件 | 作用 |
| :--- | :--- |
| `convert.py` | 导出 / trace / 转换 / 量化。已处理 eager attention、`position_ids` 常量 buffer、int32 trace、`per_channel` 量化 |
| `validate.py` | fp32 / Core ML（可选 ORT）三方余弦 + Top-5 重合 + 延迟 |
| `tokenizer.py` | Kotlin 侧 `BertTokenizer` 的 Python 镜像，**必须与线上分词完全一致**，否则测到的是分词差异 |
| `ios/EmbeddingCoreMLEngine.swift` | 可直接拷进工程的 Swift 引擎（固定 512 shape、补齐、越界防御） |
| `kmp/IosEmbeddingBridge.kt` | Kotlin/Native 桥接：Swift 注入实现，Kotlin 只做分词 + 维度校验 |

### 集成要点

1. **CLS pooling + L2 归一化已烤进模型**，Swift 侧拿到的就是 512 维单位向量，别再在端上做 pooling。
2. **协议方法名**：Kotlin 的 `initModel(...)` 在 Swift 里属于初始化器家族，实现必须写成 `doInitModel(modelData:)`。
3. **维度校验不能省**：Swift 在失败时返回空数组，若上层照单收下会写进向量索引并标记"已向量化"，此后**永不重建**，检索静默出错。

---

## 三个必然踩的坑

| 现象 | 原因 | 解法 |
| :--- | :--- | :--- |
| `only 0-dimensional arrays can be converted to Python scalars` | transformers 5.x 走 SDPA | 钉 `transformers==4.44.2` + `attn_implementation="eager"` |
| trace 报 `aten::Int(张量)` | `position_ids` 是运行时 `arange` | `register_buffer` 成常量并在 forward 显式传入 |
| `per_block quantization is supported since iOS18` | 量化粒度与最低系统版本冲突 | 用 `per_channel`（iOS16 起） |

踩坑细节与完整实测数据见作者博客（发布后在此补充链接）。本仓库只放可复现的脚本与两端接入代码。

---

## 注意

- 本仓库只覆盖 **iOS**。Android / Desktop 仍可用 ONNX Runtime，向量索引是派生数据、不参与同步，各端可独立换实现。
- 换模型后要 bump 版本号（例如 `bge-small-zh-quant-v1` → `bge-small-zh-coreml-v2`），否则旧向量不会重建。
- 许可：使用与再分发模型前请自行确认 `BAAI/bge-small-zh-v1.5` 的许可证要求。
