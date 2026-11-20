"""Kotlin KmpBertTokenizer 的 Python 镜像实现。

目的：保证转换验证时用的 token id 与 App 内 Kotlin 侧完全一致，
否则余弦相似度测的是"分词差异"而不是"模型差异"。

对应源文件：Kotlin 侧手写的 KmpBertTokenizer（BertTokenizer 的 WordPiece 实现）。
"""

from __future__ import annotations

CLS = "[CLS]"
SEP = "[SEP]"
UNK = "[UNK]"

MAX_INPUT_CHARS_PER_WORD = 200


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0xF900 <= code <= 0xFAFF)


def _is_punct(ch: str) -> bool:
    code = ord(ch)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    # Kotlin: !isLetterOrDigit && !isWhitespace
    return (not ch.isalnum()) and (not ch.isspace())


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    code = ord(ch)
    return code < 0x20 or code == 0x7F


def clean_text(text: str, do_lower_case: bool = True) -> str:
    if do_lower_case:
        text = text.lower()
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0 or code == 0xFFFD or _is_control(ch):
            continue
        if ch.isspace():
            out.append(" ")
        elif _is_cjk(ch) or _is_punct(ch):
            out.extend((" ", ch, " "))
        else:
            out.append(ch)
    return "".join(out)


def wordpiece(word: str, vocab: dict[str, int]) -> list[str]:
    if len(word) > MAX_INPUT_CHARS_PER_WORD:
        return [UNK]
    tokens: list[str] = []
    start = 0
    while start < len(word):
        end = len(word)
        cur = None
        while start < end:
            piece = word[start:end]
            if start > 0:
                piece = "##" + piece
            if piece in vocab:
                cur = piece
                break
            end -= 1
        if cur is None:
            return [UNK]
        tokens.append(cur)
        start = end
    return tokens


def tokenize(text: str, vocab: dict[str, int], do_lower_case: bool = True) -> list[str]:
    cleaned = clean_text(text, do_lower_case).strip()
    out: list[str] = []
    for word in cleaned.split():
        if word:
            out.extend(wordpiece(word, vocab))
    return out


def encode(text: str, vocab: dict[str, int], max_len: int = 512, do_lower_case: bool = True):
    """返回 (input_ids, attention_mask, token_type_ids)，与 Kotlin 版语义一致。"""
    tokens = tokenize(text, vocab, do_lower_case)
    valid = min(len(tokens), max_len - 2)
    seq_len = valid + 2
    input_ids = [vocab.get(CLS, 101)]
    for i in range(valid):
        input_ids.append(vocab.get(tokens[i], vocab.get(UNK, 100)))
    input_ids.append(vocab.get(SEP, 102))
    attention_mask = [1] * seq_len
    token_type_ids = [0] * seq_len
    return input_ids, attention_mask, token_type_ids


def load_vocab(vocab_path: str) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.rstrip("\n").strip()
            if token:
                vocab[token] = idx
    return vocab
