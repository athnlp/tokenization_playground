import gradio as gr
from playground_examples import examples
from playground_util import on_load, tokenize, tokenize_pair
from vocab import tokenizer_factory

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = JSON.stringify(Object.fromEntries(params));
        return url_params;
        }
    """

all_tokenizer_name = [
    (config.name_display, config.name_or_path)
    for config in tokenizer_factory.all_tokenizer_configs
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
        # dynamic change label
        with gr.Column():
            output_text_1 = gr.Highlightedtext(show_legend=False, show_inline_category=False)
        with gr.Column():
            output_text_2 = gr.Highlightedtext(show_legend=False, show_inline_category=False)

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
