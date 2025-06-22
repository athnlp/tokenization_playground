import json
from functools import lru_cache
from typing import Any

import gradio as gr
import pandas as pd
from playground_examples import (
    default_tokenizer_name_1,
    default_tokenizer_name_2,
    default_user_input,
)
from utils.i18n_util import get_lang
from utils.log_util import logger
from vocab import tokenizer_factory


@lru_cache
def _tokenize(text: str, tokenizer_name: str, color_num: int = 5, add_special_token: bool = False):
    logger.info(
        "param=" + json.dumps({"text": text, "tokenizer_type": tokenizer_name}, ensure_ascii=False)
    )
    pos_tokens = []
    tokenizer = tokenizer_factory.get_tokenizer(tokenizer_name)
    encoding = tokenizer.encode(text) if add_special_token else tokenizer.encode(text)
    table = []

    for idx, token_id in enumerate(encoding):
        decoded_text = tokenizer.decode([token_id])
        decoded_text = decoded_text.replace(
            " ", "⋅"
        )  # replace space with ⋅ for better visualization
        pos_tokens.extend([(decoded_text, str(idx % color_num))])

        try:
            token = tokenizer.decode([token_id])[0]
        except:
            token = {v: k for k, v in tokenizer.get_vocab().items()}[token_id]

        if isinstance(token, bytes):
            try:
                token_str = token.decode("utf-8")
            except:
                token_str = token.decode("utf-8", errors="ignore")
                logger.error(
                    f"{idx}: decode_error: "
                    + json.dumps(  # gpt_35_turbo 经常有token会decode error，这里用来记录一下
                        {
                            "tokenizer_type": tokenizer_name,
                            "token": str(token),
                            "token_str": token_str,
                        },
                        ensure_ascii=False,
                    )
                )

            # json_dumps = json.dumps(token_str)
        elif isinstance(token, str):
            token_str = token
        else:
            logger.error(
                f"{idx}: wrong type for token {token_id} {type(token)} "
                + json.dumps({"text": text, "tokenizer_type": tokenizer_name}, ensure_ascii=False)
            )
            token_str = token

        table.append({"TokenID": token_id, "Text": decoded_text})

    table_df = pd.DataFrame(table)
    logger.info(f"tokenizer_type={tokenizer_name}, Tokens={table[:4]}")
    return pos_tokens, len(encoding), table_df


def tokenize(
    text: str, tokenizer_name: str, color_num: int = 5
) -> tuple[dict[Any, Any], pd.DataFrame]:
    """Tokenize an input text."""
    pos_tokens, num_tokens, table_df = _tokenize(text, tokenizer_name, color_num)
    return gr.update(value=pos_tokens, label=f"Tokens: {num_tokens}"), table_df


def tokenize_pair(text, tokenizer_type_1, tokenizer_type_2, color_num: int = 5):
    """input_text.change."""
    pos_tokens_1, table_df_1 = tokenize(text, tokenizer_type_1, color_num)
    pos_tokens_2, table_df_2 = tokenize(text, tokenizer_type_2, color_num)
    return pos_tokens_1, table_df_1, pos_tokens_2, table_df_2


def on_load(url_params: str, request: gr.Request = None) -> tuple[str, str, str]:
    """Function triggered on page load to get URL parameters."""
    text = default_user_input
    tokenizer_type_1 = default_tokenizer_name_1
    tokenizer_type_2 = default_tokenizer_name_2
    try:
        url_params_dict = json.loads(url_params)
    except json.JSONDecodeError:
        url_params_dict = {}

    if request:
        lang, _ = get_lang(request)
        logger.info(str(request.headers))
        client_ip = request.client.host

        tokenizer_type_1 = url_params_dict.get("tokenizer1", default_tokenizer_name_1)
        tokenizer_type_2 = url_params_dict.get("tokenizer2", default_tokenizer_name_2)
        text = url_params_dict.get("text", default_user_input)
        logger.info(f"client_ip: {client_ip}; lang: {lang} params: {url_params}")
    return text, tokenizer_type_1, tokenizer_type_2
