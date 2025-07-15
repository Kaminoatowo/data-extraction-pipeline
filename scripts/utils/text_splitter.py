# src/utils/text_splitter.py
import os
from pathlib import Path
from config.config import tokenizer


def count_tokens(text):
    return len(tokenizer.encode(text))


def split_text_into_chunks(text, max_tokens):
    """
    Splits a given text into chunks based on token limits.
    Returns a list of decoded chunks.
    """
    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]


def load_and_split_text_files(folder_path: Path, max_tokens_per_chunk=1500):
    """
    Loads all .txt files in a folder and splits each into token-safe chunks.
    Returns a list of (filename, chunk_text) tuples.
    """
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".txt"))
    split_texts = []
    for f in files:
        path = folder_path / f
        with path.open("r", encoding="utf-8") as file:
            full_text = file.read()
            chunks = split_text_into_chunks(full_text, max_tokens_per_chunk)
            for idx, chunk in enumerate(chunks):
                name = f"{f} (chunk {idx + 1})" if len(chunks) > 1 else f
                split_texts.append((name, chunk))
    return split_texts
