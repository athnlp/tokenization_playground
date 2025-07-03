from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import tiktoken
from loguru import logger
from transformers import AutoTokenizer


class TokenizerImpl(Enum):
    BertTokenizer = "wordpiece.BertTokenizer"
    ByteLevelBPETokenizer = "byte_level_bpe"
    SentencePieceBPETokenizer = "sentencepiece_bpe"

    SentencePiece = auto()
    byte_level_bpe = auto()

    TikToken = auto()


@dataclass
class TokenizerConfig:
    """Tokenizer Configuration"""

    name_or_path: str
    name_display: str | None = None
    impl: TokenizerImpl | None = None
    org: str | None = None
    link: str | None = None
    desc: str | None = None
    meta: str | None = None
    level: str | None = None
    lang: str | None = None
    init_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.link is None:
            self.link = "https://huggingface.co/" + self.name_or_path
        if self.name_display is None:
            self.name_display = self.name_or_path

    @classmethod
    def init_from_json_file(cls, json_filepath: str) -> "TokenizerConfig":
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        return hash(self.name_or_path)


tokenizer_configs = [
    TokenizerConfig(
        "google-bert/bert-base-uncased",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
        desc="first add whitespace around any CJK character, then perform wordpiece tokenization.",
    ),
    TokenizerConfig(
        "google-bert/bert-base-multilingual-uncased",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
    ),
    TokenizerConfig(
        "openai-community/gpt2", impl=TokenizerImpl.SentencePiece, org="OpenAI"
    ),
    TokenizerConfig(
        "EleutherAI/gpt-neox-20b", impl=TokenizerImpl.SentencePiece, org="EleutherAI"
    ),
    TokenizerConfig(
        "Qwen/Qwen1.5-14B", impl=TokenizerImpl.SentencePiece, org="Alibaba"
    ),
    TokenizerConfig(
        "Qwen/Qwen2.5-72B", impl=TokenizerImpl.SentencePiece, org="Alibaba"
    ),
    TokenizerConfig(
        "google-t5/t5-large",
        name_display="google-t5/t5",
        impl=TokenizerImpl.SentencePiece,
        org="Google",
    ),
    TokenizerConfig("CohereForAI/aya-101", org="Cohere For AI"),
    TokenizerConfig(
        "meta-llama/Llama-3.2-3B-Instruct", impl=TokenizerImpl.SentencePiece, org="Meta"
    ),
    TokenizerConfig(
        "openai/gpt-4o",
        impl=TokenizerImpl.TikToken,
        org="OpenAI",
        link="https://github.com/openai/tiktoken",
    ),
    TokenizerConfig("google/mt5-large", org="Google"),
    TokenizerConfig("deepseek-ai/deepseek-coder-33b-instruct", org="DeepSeek"),
    TokenizerConfig("deepseek-ai/DeepSeek-V3", org="DeepSeek"),
    TokenizerConfig("ilsp/Llama-Krikri-8B-Base", org="ILSP"),
]

assert len(set([config.name_display for config in tokenizer_configs])) == len(
    tokenizer_configs
)
assert len(set([config.name_or_path for config in tokenizer_configs])) == len(
    tokenizer_configs
)
assert len(
    set([config.name_or_path.split("/")[-1] for config in tokenizer_configs])
) == len(tokenizer_configs)


class TokenizerFactory:
    def __init__(self):
        self.all_tokenizer_configs = sorted(
            tokenizer_configs, key=lambda k: k.name_display
        )
        self.all_tokenizer_names = [
            config.name_or_path for config in self.all_tokenizer_configs
        ]
        self.name_to_config_list = [
            {config.name_or_path: config for config in self.all_tokenizer_configs},
            {config.name_display: config for config in self.all_tokenizer_configs},
            {
                config.name_display.split("/")[-1]: config
                for config in self.all_tokenizer_configs
            },
        ]
        self.tokenizer_cache = {}

    def get_tokenizer_config(self, tokenizer_name: str) -> TokenizerConfig | None:
        for name_to_config in self.name_to_config_list:
            if tokenizer_name in name_to_config:
                return name_to_config[tokenizer_name]
        return None

    def get_tokenizer(self, tokenizer_name: str) -> AutoTokenizer:
        """Get the tokenizer by its name, loading it if not already cached."""
        tokenizer_config = self.get_tokenizer_config(tokenizer_name)

        if tokenizer_config in self.tokenizer_cache:
            return self.tokenizer_cache[tokenizer_config]

        tokenizer = self.load_tokenizer(tokenizer_config)

        self.tokenizer_cache[tokenizer_config] = tokenizer
        return tokenizer

    def get_name_with_hyperlink(self, tokenizer_name: str) -> str:
        def model_hyperlink(link, model_name):
            model_name = model_name
            return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

        tokenizer_config = self.get_tokenizer_config(tokenizer_name)
        return model_hyperlink(
            tokenizer_config.link, tokenizer_config.name_display.split("/")[-1]
        )

    def load_tokenizer(self, tokenizer_config):
        if tokenizer_config == None:
            print("dd")
        logger.info(f"loading tokenizer {tokenizer_config.name_or_path}")
        if (
            tokenizer_config.impl == TokenizerImpl.TikToken
            and "openai" in tokenizer_config.name_or_path
        ):
            tokenizer = tiktoken.encoding_for_model(
                tokenizer_config.name_or_path.replace("openai/", "")
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_config.name_or_path,
                trust_remote_code=True,
                **tokenizer_config.init_kwargs,
            )
        return tokenizer
