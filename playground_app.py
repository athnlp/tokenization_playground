import json
from functools import lru_cache
from typing import Any

import gradio as gr
import pandas as pd
from loguru import logger

from playground_examples import (
    default_tokenizer_name_1,
    default_tokenizer_name_2,
    default_user_input,
    examples,
)
from playground_tokenizers import TokenizerFactory


@lru_cache
def run_tokenization(
    text: str, tokenizer_name: str, color_num: int = 5, add_special_token: bool = False
) -> tuple[list[tuple[str, str]], int, pd.DataFrame]:
    """Tokenize an input text and return the tokens with their positions."""
    logger.info(
        "param="
        + json.dumps(
            {"text": text, "tokenizer_type": tokenizer_name}, ensure_ascii=False
        )
    )
    pos_tokens = []
    tokenizer = TokenizerFactory().get_tokenizer(tokenizer_name)
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
                    f"{idx}: decode_error: {tokenizer_name}, {token} {token_str}"
                )

        elif isinstance(token, str):
            token_str = token
        else:
            logger.error(
                f"{idx}: wrong type for token {token_id} {type(token)} "
                + json.dumps(
                    {"text": text, "tokenizer_type": tokenizer_name}, ensure_ascii=False
                )
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
    pos_tokens, num_tokens, table_df = run_tokenization(text, tokenizer_name, color_num)
    return gr.update(value=pos_tokens, label=f"Tokens: {num_tokens}"), table_df


def tokenize_pair(
    text: str, tokenizer_name_1: str, tokenizer_name_2: str, color_num: int = 5
):
    """input_text.change."""
    pos_tokens_1, table_df_1 = tokenize(
        text=text, tokenizer_name=tokenizer_name_1, color_num=color_num
    )
    pos_tokens_2, table_df_2 = tokenize(
        text=text, tokenizer_name=tokenizer_name_2, color_num=color_num
    )
    return pos_tokens_1, table_df_1, pos_tokens_2, table_df_2


def on_load(url_params: str, request: gr.Request | None = None) -> tuple[str, str, str]:
    """Function triggered on page load to get URL parameters."""
    text = default_user_input
    tokenizer_type_1 = default_tokenizer_name_1
    tokenizer_type_2 = default_tokenizer_name_2
    return text, tokenizer_type_1, tokenizer_type_2


get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = JSON.stringify(Object.fromEntries(params));
        return url_params;
        }
    """

all_tokenizer_name = [
    (config.name_display, config.name_or_path)
    for config in TokenizerFactory().all_tokenizer_configs
]

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("## Input Text")
        dropdown_examples = gr.Dropdown(
            sorted(examples.keys()),
            value="Examples",
            type="index",
            allow_custom_value=True,
            show_label=False,
            container=False,
            scale=0,
            elem_classes="example-style",
        )
    user_input = gr.Textbox(
        label="Input Text",
        lines=5,
        show_label=False,
    )

    with gr.Row():
        with gr.Column(scale=6), gr.Group():
            tokenizer_name_1 = gr.Dropdown(all_tokenizer_name, label="Tokenizer 1")

        with gr.Column(scale=6), gr.Group():
            tokenizer_name_2 = gr.Dropdown(all_tokenizer_name, label="Tokenizer 2")

    with gr.Row():
        with gr.Column():
            output_text_1 = gr.Highlightedtext(
                show_legend=False, show_inline_category=False
            )
        with gr.Column():
            output_text_2 = gr.Highlightedtext(
                show_legend=False, show_inline_category=False
            )

    with gr.Row():
        output_table_1 = gr.Dataframe()
        output_table_2 = gr.Dataframe()

    tokenizer_name_1.change(
        tokenize, [user_input, tokenizer_name_1], [output_text_1, output_table_1]
    )

    tokenizer_name_2.change(
        tokenize, [user_input, tokenizer_name_2], [output_text_2, output_table_2]
    )

    user_input.change(
        tokenize_pair,
        [user_input, tokenizer_name_1, tokenizer_name_2],
        [output_text_1, output_table_1, output_text_2, output_table_2],
        show_api=False,
    )

    dropdown_examples.change(
        lambda example_idx: (
            examples[sorted(examples.keys())[example_idx]]["text"],
            examples[sorted(examples.keys())[example_idx]]["tokenizer_1"],
            examples[sorted(examples.keys())[example_idx]]["tokenizer_2"],
        ),
        dropdown_examples,
        [user_input, tokenizer_name_1, tokenizer_name_2],
        show_api=False,
    )

    demo.load(
        fn=on_load,
        inputs=[user_input],
        outputs=[user_input, tokenizer_name_1, tokenizer_name_2],
        js=get_window_url_params,
        show_api=False,
    )

if __name__ == "__main__":
    demo.launch(share=True)
