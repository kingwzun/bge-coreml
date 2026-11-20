"""把 BGE-Small-ZH 转换成 Core ML 模型（.mlmodel），可选权重量化。

用法：
    python convert.py --model-dir ./bge-small-zh --out ./out/bge_int8.mlpackage --quant int8
    python convert.py --model-dir ./bge-small-zh --out ./out/bge_fp32.mlpackage --quant none

设计要点（与 App 内现有实现严格对齐）：
- 输入 input_ids / attention_mask / token_type_ids，shape 固定 (1, SEQ)，int32
- 输出 sentence_embedding，shape (1, 512)：CLS pooling + L2 归一化已烤进图里
  （这是关键，Swift 侧才能和现状一样"只取一个 512 维数组"）
- mlprogram + iOS17 起，才能上 ANE
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import BertModel


class BgeEncoder(torch.nn.Module):
    """BERT 编码器 + CLS pooling + L2 归一化，输出句向量。"""

    def __init__(self, bert: BertModel, pooling: str = "cls", seq_len: int = 512):
        super().__init__()
        self.bert = bert
        self.pooling = pooling
        # 关键：position_ids 用常量 buffer。
        # 否则 BERT 内部会走 torch.arange(seq_length) 的动态路径，trace 出来是
        # aten::Int(张量) —— coremltools 直接报 "only 0-dimensional arrays..."。
        self.register_buffer(
            "position_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=self.position_ids,
            return_dict=False,
        )[0]  # last_hidden_state: (1, seq, 512)

        if self.pooling == "cls":
            v = out[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(out.dtype)
            v = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return F.normalize(v, p=2, dim=1)


def build_model(model_dir: str, pooling: str, seq_len: int = 512) -> BgeEncoder:
    # attn_implementation="eager" 是必须的：
    # transformers 5.x 默认走 SDPA / masking_utils，其内部的 int 索引运算无法被
    # coremltools 转换（报 "only 0-dimensional arrays can be converted to Python scalars"）。
    # eager 路径是经典 attention，也是 Apple 官方教程用的路径。
    bert = BertModel.from_pretrained(model_dir, attn_implementation="eager").eval()
    return BgeEncoder(bert, pooling, seq_len).eval()


def convert(model: BgeEncoder, seq_len: int, deployment: str) -> ct.models.MLModel:
    # 关键：用 int32 trace。若用 int64 trace 而输入声明成 int32，
    # coremltools 会在图里插入 int64->int32 的 cast，某些版本直接报错。
    ids = torch.ones((1, seq_len), dtype=torch.int32)
    mask = torch.ones((1, seq_len), dtype=torch.int32)
    types = torch.zeros((1, seq_len), dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(model, (ids, mask, types))

    target = {
        "iOS17": ct.target.iOS17,
        "iOS16": ct.target.iOS16,
    }[deployment]

    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="token_type_ids", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="sentence_embedding", dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
    )


def quantize(model: ct.models.MLModel, mode: str, deployment: str) -> ct.models.MLModel:
    if mode == "none":
        return model

    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
        palettize_weights,
    )

    if mode == "int8":
        # 8bit 线性对称量化，per-block(32) 是 iOS17 上精度/体积平衡最好的档
        cfg = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                # per_block 需要 iOS18；本项目最低 iOS15.6/16.6，用 per_channel（iOS16 起支持）
                mode="linear_symmetric", dtype="int8", granularity="per_channel"
            )
        )
        return linear_quantize_weights(model, cfg)

    if mode == "int4":
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4))
        return palettize_weights(model, cfg)

    raise ValueError(f"未知量化模式: {mode}")


def disk_size(path: str) -> float:
    if os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / 1048576
    return os.path.getsize(path) / 1048576


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./bge-small-zh")
    ap.add_argument("--out", required=True, help="输出路径，建议 .mlpackage")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    ap.add_argument("--quant", default="none", choices=["none", "int8", "int4"])
    ap.add_argument("--deployment", default="iOS16", choices=["iOS17", "iOS16"])
    args = ap.parse_args()

    t0 = time.time()
    model = build_model(args.model_dir, args.pooling, args.seq_len)
    print(f"[1/3] PyTorch 模型就绪 ({time.time()-t0:.1f}s)")

    mlmodel = convert(model, args.seq_len, args.deployment)
    print(f"[2/3] Core ML 转换完成 ({time.time()-t0:.1f}s)")

    if args.quant != "none":
        mlmodel = quantize(mlmodel, args.quant, args.deployment)
        print(f"[3/3] 量化完成: {args.quant}")

    if os.path.exists(args.out):
        import shutil
        if os.path.isdir(args.out):
            shutil.rmtree(args.out)
        else:
            os.remove(args.out)

    mlmodel.save(args.out)
    print(f"✅ 已保存: {args.out}  ({disk_size(args.out):.1f}MB, 总耗时 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
