import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from utils.lang_util import detect_language_by_unicode, language_ranges
from utils.log_util import logger
from utils.text_util import contains_digit, get_space_count
from vocab import tokenizer_factory

CURRENT_DIR = Path.parent(Path.resolve(__file__))

cache = {}
default_columns = ["digit", "zh"]


def text_to_unicode(text: str) -> str:
    """Convert text to unicode representation."""
    return "".join(rf"\u{ord(character):04X}" for character in text)


def calculate_dist(token_lens: list[int]) -> str:
    """Calculate the distribution of token lengths."""
    if not token_lens:
        return "-"
    return f"{min(token_lens)},{round(np.median(token_lens))},{max(token_lens)}"


def iter_vocab(
    tokenizer_name: str,
    from_cache: bool = True,
    cache_dir: str = "stats",
) -> pd.DataFrame | dict:
    """:param tokenizer_name:
    :param from_cache:
    :param cache_dir:
    :return:
    """
    tokenizer_config = tokenizer_factory.get_tokenizer_config(tokenizer_name)

    cache_dir = os.path.join(CURRENT_DIR, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # load from cache
    cache_path = os.path.join(cache_dir, "character_stats.json")
    if not cache and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f_tmp:
            cache.update(json.load(f_tmp))
    if from_cache and tokenizer_name in cache:
        # logger.info(f"load {tokenizer_config.name_or_path} from cache")
        return cache[tokenizer_name]

    tokenizer = tokenizer_factory.get_tokenizer(tokenizer_name)

    tokens_by_lang = {lang[1]: [] for lang in language_ranges}
    digit_tokens = []
    space_tokens = []
    byte_tokens = []

    buffer = []
    for token_id in range(tokenizer.vocab_size):
        # for token_id in tokenizer.get_vocab():
        # for token_id in range(len(tokenizer)):
        decode_str = tokenizer.decode([token_id], skip_special_tokens=False)
        token = tokenizer.convert_ids_to_tokens([token_id], skip_special_tokens=False)[0]
        tags = []
        if token is None:  # 有些词典有空的id（不连续）
            continue
        if isinstance(token, bytes):
            token = token.decode("utf-8", errors="ignore")

        if hasattr(tokenizer, "sp_model") and tokenizer.sp_model.is_byte(token_id):
            tags.append("is_byte")
            byte_tokens.append(token)

        language_tags = detect_language_by_unicode(decode_str)
        for language in language_tags:
            tokens_by_lang[language[1]].append(decode_str)

        if contains_digit(decode_str):
            tags.append("digit")
            digit_tokens.append(decode_str)

        space_count = get_space_count(decode_str)
        if space_count > 0:
            space_tokens.append(decode_str)

        buffer.append(
            json.dumps(
                {
                    "id": token_id,
                    "token": token,
                    "token_decode": decode_str,
                    "token_dumps": json.dumps(token),
                    "token_unicode": text_to_unicode(token),
                    "token_len": len(decode_str),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    result = {
        "tokenizer": tokenizer_factory.get_name_with_hyperlink(tokenizer_name),
        "organization": tokenizer_config.org,
        "vocab_size": len(tokenizer),
        "num(digit)": len(digit_tokens),
        "len(digit)": calculate_dist([len(token) for token in digit_tokens]),
        "num(space)": len(space_tokens),
        "len(space)": calculate_dist([len(token) for token in space_tokens]),
    }

    for lang, tokens in tokens_by_lang.items():
        result[f"num({lang})"] = len(tokens)
        result["len(" + lang + ")"] = calculate_dist([len(token) for token in tokens])

    out_path = os.path.join(
        cache_dir, f"iter_vocab/{tokenizer_name.replace('/', '_')}.vocab.jsonl"
    )
    with open(out_path, "w", encoding="utf-8") as f_out:
        for line in buffer:
            f_out.write(line)
    len_before = len(cache)
    cache[tokenizer_name] = result
    len_after = len(cache)
    logger.info(f"saving {tokenizer_name} to memory and file cache: {len_before}->{len_after}")
    with open(cache_path, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps(cache, ensure_ascii=False, indent=2))
    return result


def to_dataframe(stats: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    table = []
    for stat in stats.values():
        filtered_stat = {}
        for k, v in stat.items():
            if not k.startswith("num") and not k.startswith("len"):
                filtered_stat[k] = v
            if any(column in k for column in columns):
                k = k.replace("ja-kana", "kana")
                filtered_stat[k] = v
        table.append(filtered_stat)
    return pd.DataFrame(table)


def get_character_table(
    tokenizer_filter: str | None = None,
    columns: list | None = None,
    return_type: Literal["dict", "dataframe"] | None = "dataframe",
) -> pd.DataFrame | dict:
    logger.info(f"columns: {columns}, tokenizer_filter: {tokenizer_filter}")
    stats = {}
    if columns is None:
        columns = default_columns
    if tokenizer_filter is not None:
        tokenizer_names = [
            tokenizer_config.name_or_path
            for tokenizer_config in tokenizer_factory.all_tokenizer_configs
            if tokenizer_filter.lower() in tokenizer_config.name_or_path.lower()
        ]
    else:
        tokenizer_names = tokenizer_factory.all_tokenizer_names

    for tokenizer_name in tokenizer_names:
        stat = iter_vocab(tokenizer_name)
        stats[tokenizer_name] = stat

    if return_type == "dataframe":
        stats = to_dataframe(stats, columns)
    return stats


if __name__ == "__main__":
    # aa = get_character_table(tokenizer_filter="baichuan")
    df = get_character_table()
    logger.info(f"\n{df.to_markdown(index=False)}")
