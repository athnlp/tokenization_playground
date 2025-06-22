import os

import gradio as gr
from huggingface_hub import login

from playground_app import demo as playground_tab

auth_token = os.environ.get("HF_TOKEN", None)
if auth_token:
    login(token=auth_token)


title = """
<div align="center">
    <span>Tokenization Playground</span>
</div>
"""

with gr.Blocks() as demo:
    gr.HTML(f"<h1 style='text-align: center; margin-bottom: 1rem'>{title}</h1>")
    playground_tab.render()

if __name__ == "__main__":
    demo.launch(share=True)
