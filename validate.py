"""数值一致性 + 延迟校验：Core ML 模型 vs PyTorch fp32（真值）vs 可选 .ort 现状。

用法：
    python validate.py --cm ./out/bge_int8.mlpackage \
                        --model-dir ./bge-small-zh \
                        --vocab ./vocab.txt
    # 想同时对比旧实现（可选）：
    python validate.py --cm ./out/bge_int8.mlpackage --vocab ./vocab.txt --ort ./old.onnx.quant.ort

判定标准：
- CoreML 与 PyTorch fp32 的余弦 >= 0.99 → 新引擎合格
- 同时看 Top-5 近邻是否与 fp32 一致（余弦高但排序乱，说明量化伤了关键维度）
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch
from transformers import BertModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokenizer as tk  # noqa: E402

SEQ = 512

TEXTS = [
    "帮我建一个每天阅读20分钟的习惯",
    "晨跑打卡，今天跑了五公里",
    "我的收藏里有三只盲盒，都是限量款",
    "2026年11月19日 苏州 出差报销单",
    "会议记录：讨论客户端 agent 架构改造",
    "买了新的机械键盘，红轴",
    "下个月要交房租，记得提前转账",
    "读完《人类简史》第三章",
    "照片：外婆生日蛋糕，三层水果味",
    "这个模型推理太慢了，要换成 Core ML",
    "weekend plan: hiking in Suzhou",
    "发票 No.0321 金额 1,280 元",
]


def pad(seq: list[int]) -> np.ndarray:
    arr = np.zeros((1, SEQ), dtype=np.int32)
    arr[0, : len(seq)] = seq
    return arr


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def top5(vec: np.ndarray, mat: np.ndarray) -> list[int]:
    scores = mat @ vec
    return list(np.argsort(-scores)[:5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm", required=True, help="Core ML .mlpackage / .mlmodelc")
    ap.add_argument("--model-dir", default="./bge-small-zh", help="fp32 参考模型目录")
    ap.add_argument("--vocab", required=True, help="vocab.txt 路径")
    ap.add_argument("--ort", default=None, help="可选：旧 .ort 模型，用于对比现状")
    ap.add_argument("--runs", type=int, default=20, help="延迟测试次数")
    args = ap.parse_args()

    vocab = tk.load_vocab(args.vocab)

    bert = BertModel.from_pretrained(args.model_dir, attn_implementation="eager").eval()

    import coremltools as ct
    cm_model = ct.models.MLModel(args.cm)

    sess = None
    if args.ort:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.ort, providers=["CPUExecutionProvider"])

    def run_torch(text: str) -> np.ndarray:
        ids, mask, types = tk.encode(text, vocab)
        with torch.no_grad():
            h = bert(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.tensor([mask]),
                token_type_ids=torch.tensor([types]),
            ).last_hidden_state[0]
        return torch.nn.functional.normalize(h[0], p=2, dim=0).numpy()

    def run_cm(text: str) -> np.ndarray:
        ids, mask, types = tk.encode(text, vocab)
        out = cm_model.predict({
            "input_ids": pad(ids),
            "attention_mask": pad(mask),
            "token_type_ids": pad(types),
        })["sentence_embedding"]
        return np.array(out).ravel()

    def run_ort(text: str) -> np.ndarray:
        ids, mask, types = tk.encode(text, vocab)
        out = sess.run(["sentence_embedding"], {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
            "token_type_ids": np.array([types], dtype=np.int64),
        })[0]
        return out[0]

    print(f"\n=== 数值一致性（{len(TEXTS)} 条文本）===")
    ref = np.stack([run_torch(t) for t in TEXTS])          # (N, 512) fp32 真值
    cm_vecs = [run_cm(t) for t in TEXTS]
    ort_vecs = [run_ort(t) for t in TEXTS] if sess else None

    def stat(name, vals):
        print(f"{name:24s} mean={statistics.mean(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

    stat("CoreML vs fp32", [cos(c, r) for c, r in zip(cm_vecs, ref)])
    if ort_vecs:
        stat("ORT vs fp32", [cos(o, r) for o, r in zip(ort_vecs, ref)])
        stat("CoreML vs ORT", [cos(c, o) for c, o in zip(cm_vecs, ort_vecs)])

    # Top-5 近邻是否一致（这是"检索有没有变差"的直接指标）
    ref_hits = [len(set(top5(c, ref)) & set(top5(r, ref))) for c, r in zip(cm_vecs, ref)]
    print(f"{'CoreML Top-5 重合':24s} mean={statistics.mean(ref_hits):.2f} / 5")
    if ort_vecs:
        ort_hits = [len(set(top5(o, ref)) & set(top5(r, ref))) for o, r in zip(ort_vecs, ref)]
        print(f"{'ORT Top-5 重合':24s} mean={statistics.mean(ort_hits):.2f} / 5")

    def bench(fn, runs):
        for _ in range(3):
            fn(TEXTS[0])
        t0 = time.perf_counter()
        for i in range(runs):
            fn(TEXTS[i % len(TEXTS)])
        return (time.perf_counter() - t0) / runs * 1000

    print(f"\n=== 单条延迟（{args.runs} 次均值）===")
    print(f"Core ML : {bench(run_cm, args.runs):.1f} ms")
    if sess:
        print(f"ORT     : {bench(run_ort, args.runs):.1f} ms")
    print(f"PyTorch : {bench(run_torch, args.runs):.1f} ms")


if __name__ == "__main__":
    main()
