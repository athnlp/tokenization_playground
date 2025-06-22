from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict

import tiktoken
from transformers import AutoTokenizer
from utils.log_util import logger

"""Interface:
# https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py

tokenizer.encode -> List[int]: Converts a string to a sequence of ids (integer)
tokenizer.decode
    tokenizer.convert_tokens_to_string   # gpt4 没有这个方法
tokenizer.convert_ids_to_tokens
tokenizer.tokenize -> List[str]:  Converts a string into a sequence of tokens ->


tokenizer.parent = ""
tokenizer.vocab_size   
tokenizer.get_vocab()   # gpt-neox-20b, llama
tokenizer.type = TokenizerType.ByteBPE.name
tokenizer.implementation = TokenizerImpl.SentencePiece.name   # https://github.com/facebookresearch/llama/blob/main/llama/tokenizer.py
  "HFGPT2Tokenizer", "HFTokenizer", "GPT2BPETokenizer", "CharLevelTokenizer", "TiktokenTokenizer", "SPMTokenizer", https://github.com/EleutherAI/gpt-neox/blob/main/tools/preprocess_data.py

    
tokenizer.comments = "split all numbers into individual digits, " \
                     "and fallback to bytes to decompose unknown UTF-8 characters"

tokenizer.all_special_tokens  # baichuan
tokenizer.special_tokens_set   # gpt3.5_turbo
tokenizer.special_tokens_map   
"""


class TokenizerImpl(Enum):
    """
    - https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/__init__.py
    - https://huggingface.co/docs/transformers/tokenizer_summary
    - https://github.com/EleutherAI/gpt-neox/blob/main/megatron/tokenizer/tokenizer.py

    ## google/BertTokenizer
    - https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/bert_wordpiece.py
    - 特征
        - 算法：BERT的编码器是 BPE-WordPiece，将单词拆分成多个前缀符号（比如BERT中的##）最小单元
        - 词典：有##开头的token，表示subword，
            - 中文采用char粒度分词
            - 英文采用  WordPiece




    ## google/sentencepiece
    - https://github.com/google/sentencepiece/blob/3863f7648e5d8edb571ac592f3ac4f5f0695275a/src/sentencepiece_model.proto#L48
    - 支持 sentencepiece 和 wordpiece
        - sentencepiece 有byte-bpe吗？
            - UNIGRAM = 1;  // Unigram language model with dynamic algorithm
            - BPE = 2;      // Byte Pair Encoding
            - WORD = 3;     // Delimitered by whitespace.
            - CHAR = 4;     // tokenizes into character sequence
        - wordpiece
    - 特征：
        - 训练: spm_train --model_type unigram/bpe/char/word
        - 特殊符号： Ġ
        - 文件: *.sp_model  或 *.model  (可选文件 .vocab，) spm简称   (其他格式比如 tokenizer.json是给hf_tokenizer兼容用的)
        - 实现:
            - 依赖: protobuf
            - 训练: `import sentencepiece as spm; spm.SentencePieceTrainer.train` 或 `spm_train`
            - 加载: `import sentencepiece as spm; spm.SentencePieceProcessor().Load(vocab_file)`
            - 方法: 是SentencePieceProcessor类型，sp_model.id_to_piece，有tokenizer.json tokenizer.model，
            - 分词:
                - pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False)
        - 词典:  词典字符有 ▁  (U+2581) ，表示空格或句首。
    - 示例：google-t5, llama，baichuan, orion,
        - llama: tokenizer.json(包含model.vocab model.merges)  tokenizer.model
        - grok: 原始是 .model文件，后面转成了 tokenizer.json
        - google-t5: tokenizer.json, spiece.model
        - Skywork-13B-Math: tokenizer.model
        - xlm_roberta: sentencepiece.bpe.model
        - GPT2Tokenizer
            - tokenizer.json, vocab.json, merges.txt   (https://huggingface.co/openai-community/gpt2)
            - vocab.bpe, encoder.json, dict.txt  （fairseq版本，不常用，可以忽略这个版本）



    ## thu/icetk
      - icetk： sentencepiece的分支，支持image_tokenizer。
    - glm, chatglm1, chatglm2

    ## huggingface/tokenizers
    - https://github.com/huggingface/tokenizers
    - VS sentencepiece
        - 支持sentencepiece
            - .model转化为 (merges.txt + vocab.json) 或者 tokenizer.json
                - https://github.com/huggingface/tokenizers/blob/main/bindings/python/scripts/sentencepiece_extractor.py
            - 加载 merges.txt, vocab.json
                - SentencePieceBPETokenizer  https://github.com/huggingface/tokenizers/blob/v0.19.1/bindings/python/py_src/tokenizers/implementations/sentencepiece_bpe.py#L10
        - 在 sentencepiece基础上，hf_tokenizer支持pre-tokenization的正则表达式，对tab和换行支持更好，支持special token
    - 类型： 支持 BBPE, WordPiece or Unigram
    - 特征：
        - 文件: tokenizer.json(包含后两个文件的内容), merges.txt, vocab.json
            - added_tokens 在vocab中不一定存在。
        - 实现:
            - 训练: `from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer`
            - 加载:
            - 方法: .model.from_file  .model.save   .model.token_to_id  .model.tokenize
        - .model 是 tokenizer.models.BPE 类型
        - 词典有 Ġ  "\u0120" 开头
        - 优势
        -
    - 示例：gpt2, gpt_neox_20b, moss, bloom, qwen2
    - 优势：相对sentence piece，
        - ss

    ## openai/tiktoken
    - 特征：空格就是空格，
    - 示例：gpt3.5 gpt4, qwen,
    """

    """ 算法体系  https://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/huggingface_transformer/chapters/1_tokenizer.html
    - word-base tokenizer:
    - char-base tokenizer:
    - subword-based Tokenizer
        - BPE 
            - byte-bpe: base vocabulary大小是256
        - WordPiece:
            - 相比BPE，WordPiece 仅保存最终词表，而不保存学到的 merge rule
        - Unigram
    - SentencePiece
    
    """

    # 分类体系：https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/
    BertTokenizer = "wordpiece.BertTokenizer"
    JapaneseTokenizer = (
        "wordpiece.MecabTokenizer",
        "https://github.com/polm/fugashi",
    )  # 常用日语包 ipadic，fugashi，
    ByteLevelBPETokenizer = "byte_level_bpe"  # BBPE
    SentencePieceBPETokenizer = "sentencepiece_bpe"

    # 分类体系

    # SentencePeice(BPE)
    SentencePiece = auto()  # sentencepiece.bpe, sentencepiece.unigram, sentencepiece.char, sentencepiece.word,
    byte_level_bpe = auto()
    # HFTokenizer = auto()  # , 支持
    TikToken = auto()
    # subword-nmt
    # WordPiece


# load_vocab_with_SPECIAL_TOKEN = True # 如果不包含会导致计算词典大小错误、overlap_token计算不一致。


@dataclass
class TokenizerConfig:
    """
    https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/blob/main/src/leaderboard/read_evals.py
    """

    name_or_path: str  # org/model (path on hub), as unique id
    name_display: str = None  #
    impl: TokenizerImpl = None  # implementation, tokenizer_class/type
    org: str = None
    link: str = None  # http://**
    desc: str = None  # description
    meta: str = None
    level: str = None  # char-level, word-level, byte-level
    lang: str = None
    init_kwargs: Dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self):
        if self.link is None:
            self.link = "https://huggingface.co/" + self.name_or_path  # TODO + revision
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


# TODO: append link and description to the end of dropdown button.
# Add tokenizer_class/type, comments
_all_tokenizer_config = [
    # bert style tokenizers
    TokenizerConfig(
        "google-bert/bert-base-cased",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
        desc="first add whitespace around any CJK character, then perform wordpiece tokenization.",
    ),
    TokenizerConfig(
        "google-bert/bert-base-uncased",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
        desc="first add whitespace around any CJK character, then perform wordpiece tokenization.",
    ),
    TokenizerConfig(
        "google-bert/bert-base-chinese",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
        desc="first add whitespace around any CJK character, then perform wordpiece tokenization.",
    ),
    TokenizerConfig(
        "google-bert/bert-base-german-cased",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
    ),
    TokenizerConfig(
        "dbmdz/bert-base-german-uncased", impl=TokenizerImpl.BertTokenizer, org="dbmdz"
    ),
    TokenizerConfig(
        "asafaya/bert-base-arabic", impl=TokenizerImpl.BertTokenizer, org="-"
    ),
    TokenizerConfig(
        "google-bert/bert-base-multilingual-uncased",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
    ),
    TokenizerConfig(
        "google-bert/bert-base-multilingual-cased",
        impl=TokenizerImpl.BertTokenizer,
        org="Google",
    ),
    TokenizerConfig(
        "tohoku-nlp/bert-base-japanese",
        impl=TokenizerImpl.BertTokenizer,
        org="Tohoku",
        desc="The texts are first tokenized by MeCab morphological parser with the IPA dictionary, "
        "then split into subwords by the WordPiece algorithm.",
    ),
    TokenizerConfig(
        "clue/roberta_chinese_clue_tiny",
        name_display="clue/roberta-chinese-clue",
        impl=TokenizerImpl.BertTokenizer,
        org="CLUE",
        init_kwargs={"revision": "refs/pr/1"},
        desc="",
        meta="去掉了繁体字, https://github.com/CLUEbenchmark/CLUEPretrainedModels/blob/master/README.md",
    ),
    TokenizerConfig(
        "eson/kplug-base-encoder",
        name_display="eson/kplug",
        impl=TokenizerImpl.BertTokenizer,
        org="JD",
    ),
    TokenizerConfig(
        "ckiplab/gpt2-base-chinese", impl=TokenizerImpl.BertTokenizer, org="SINICA"
    ),  # 台湾中央研究院
    # WoBERT  https://kexue.fm/archives/7758
    # WoBERT Plus  https://github.com/ZhuiyiTechnology/WoBERT
    # gpt2 style tokenizers
    TokenizerConfig(
        "openai-community/gpt2", impl=TokenizerImpl.SentencePiece, org="OpenAI"
    ),
    # byte-level BPE,没有byte，是unicode-level的吗？
    TokenizerConfig(
        "ClassCat/gpt2-base-french", impl=TokenizerImpl.SentencePiece, org="ClassCat"
    ),
    TokenizerConfig(
        "ClassCat/gpt2-base-spanish", impl=TokenizerImpl.SentencePiece, org="ClassCat"
    ),
    TokenizerConfig(
        "fnlp/moss-moon-003-sft",
        impl=TokenizerImpl.SentencePiece,
        init_kwargs={"revision": "refs/pr/6"},
        org="Fudan",
        desc="This tokenizer has been trained to treat spaces like parts of the tokens "
        "(a bit like sentencepiece) so a word will be encoded differently whether "
        "it is at the beginning of the sentence (without space) or not",
        meta="在gpt2词典基础上，扩充了5万中文",
    ),
    TokenizerConfig(
        "bigscience/bloom",
        impl=TokenizerImpl.SentencePiece,
        org="BigScience",
        meta="比gpt_neox的词典 对中文支持更好。",
    ),
    # ("bloomz_6b4_zh",
    # ("BelleGroup/BELLE-7B-2M",   # 模型和词典都基于bloom
    #
    TokenizerConfig(
        "EleutherAI/gpt-neox-20b", impl=TokenizerImpl.SentencePiece, org="EleutherAI"
    ),  # 5万
    TokenizerConfig(
        "cyberagent/open-calm-7b", impl=TokenizerImpl.SentencePiece, org="CyberAgent"
    ),  # GPTNeoXTokenizer
    TokenizerConfig(
        "abeja/gpt-neox-japanese-2.7b", impl=TokenizerImpl.SentencePiece, org="ABEJA"
    ),
    TokenizerConfig(
        "rinna/bilingual-gpt-neox-4b",
        impl=TokenizerImpl.SentencePiece,
        org="ABEJA",
        lang="en/ja",
    ),
    TokenizerConfig(
        "Qwen/Qwen1.5-14B", impl=TokenizerImpl.SentencePiece, org="Alibaba"
    ),  # 15万，速度有点慢
    TokenizerConfig(
        "Qwen/Qwen1.5-110B", impl=TokenizerImpl.SentencePiece, org="Alibaba"
    ),
    TokenizerConfig(
        "Qwen/Qwen1.5-1.8B", impl=TokenizerImpl.SentencePiece, org="Alibaba"
    ),
    TokenizerConfig("Qwen/Qwen2-0.5B", impl=TokenizerImpl.SentencePiece, org="Alibaba"),
    TokenizerConfig("Qwen/Qwen2-72B", impl=TokenizerImpl.SentencePiece, org="Alibaba"),
    TokenizerConfig(
        "Qwen/Qwen2.5-0.5B", impl=TokenizerImpl.SentencePiece, org="Alibaba"
    ),
    TokenizerConfig(
        "Qwen/Qwen2.5-72B", impl=TokenizerImpl.SentencePiece, org="Alibaba"
    ),
    TokenizerConfig(
        "HuggingFaceH4/starchat-alpha", impl=TokenizerImpl.SentencePiece, org="-"
    ),
    ####### google/sentencepiece tokenizer:
    # T5 llama internlm
    TokenizerConfig(
        "google-t5/t5-large",
        name_display="google-t5/t5",
        impl=TokenizerImpl.SentencePiece,
        org="Google",
    ),
    # t5_small, t5_base, t5_large, flan_t5_base,
    # ("t5_base", "", "sentencepiece"),
    # TokenizerConfig("google/flan-t5-base", impl=TokenizerImpl.SentencePiece, ),
    TokenizerConfig(
        "lmsys/fastchat-t5-3b-v1.0",
        impl=TokenizerImpl.SentencePiece,
        org="LMSYS",
        init_kwargs={
            "use_fast": False
        },  # 解决 pyo3_runtime.PanicException: AddedVocabulary bad split
    ),
    TokenizerConfig(
        "CohereForAI/aya-101", org="Cohere For AI"
    ),  # "tokenizer_class": "T5Tokenizer",
    TokenizerConfig(
        "ClueAI/ChatYuan-large-v2", impl=TokenizerImpl.SentencePiece, org="CLUE"
    ),
    TokenizerConfig(
        "ClueAI/PromptCLUE-base", impl=TokenizerImpl.SentencePiece, org="CLUE"
    ),
    # byte-level BPE
    # '中文单字': 700, '中文多字': 0  meta-llama/Meta-Llama-3.1-405B
    #
    TokenizerConfig(
        "meta-llama/Llama-3.2-1B-Instruct", impl=TokenizerImpl.SentencePiece, org="Meta"
    ),
    TokenizerConfig(
        "meta-llama/Llama-3.2-3B-Instruct", impl=TokenizerImpl.SentencePiece, org="Meta"
    ),
    # TokenizerConfig("meta-llama/Llama-3.3-70B-Instruct", impl=TokenizerImpl.SentencePiece,
    #                 org="Meta"),
    TokenizerConfig(
        "meta-llama/Meta-Llama-3.1-405B", impl=TokenizerImpl.SentencePiece, org="Meta"
    ),
    TokenizerConfig(
        "NousResearch/Hermes-3-Llama-3.1-405B",
        impl=TokenizerImpl.SentencePiece,
        org="NousResearch",
    ),
    TokenizerConfig(
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        name_display="Meta/llama3",
        impl=TokenizerImpl.SentencePiece,
        org="Meta",
        desc="llama split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters",
    ),
    TokenizerConfig(
        "NousResearch/Llama-2-7b-chat-hf",
        name_display="Meta/llama2",
        impl=TokenizerImpl.SentencePiece,
        org="Meta",
    ),
    TokenizerConfig(
        "huggyllama/llama-7b",
        name_display="Meta/llama",
        impl=TokenizerImpl.SentencePiece,
        org="Meta",
    ),
    TokenizerConfig(
        "hpcai-tech/grok-1",
        name_display="xai-org/grok-1",
        impl=TokenizerImpl.SentencePiece,
        org="xAI",
    ),
    # 由.model文件转化为了
    TokenizerConfig(
        "hfl/chinese-llama-lora-7b",
        impl=TokenizerImpl.SentencePiece,
        org="-",
        meta="向原始LLaMA的词汇表中添加2w个中文词汇，针对原版LLaMA模型扩充了中文词表， 提升了中文编解码效率",
    ),
    #
    TokenizerConfig(
        "hfl/chinese-llama-2-7b",
        impl=TokenizerImpl.SentencePiece,
        org="-",
        meta="重新设计了新词表（大小：55296），进一步提升了中文字词的覆盖程度",
    ),  #
    TokenizerConfig(
        "hfl/llama-3-chinese-8b", impl=TokenizerImpl.SentencePiece, org="-"
    ),
    TokenizerConfig(
        "hfl/chinese-alpaca-lora-7b", impl=TokenizerImpl.SentencePiece, org="-"
    ),
    # 中文Alpaca模型在上述中文LLaMA模型的基础上进一步使用了指令数据进行精调。  "比chinese_llama词典多一个`[PAD]`，请勿混用"
    #
    # ("belle_llama_ext_7b",
    # ("alpaca_7b",
    TokenizerConfig(
        "baichuan-inc/Baichuan-7B",
        name_display="baichuan-inc/baichuan",
        impl=TokenizerImpl.SentencePiece,
        level="byte-level",
        org="Baichuan",
    ),
    TokenizerConfig(
        "baichuan-inc/Baichuan2-7B-Chat",
        name_display="baichuan-inc/baichuan2",
        impl=TokenizerImpl.SentencePiece,
        org="Baichuan",
        desc="expand the vocabulary size from 64000 in Baichuan1 to 125696",
    ),
    TokenizerConfig(
        "internlm/internlm-chat-7b",
        impl=TokenizerImpl.SentencePiece,
        org="Shanghai AI Lab",
    ),
    # 上海AI实验室 +  商汤
    TokenizerConfig(
        "internlm/internlm2-chat-7b",
        impl=TokenizerImpl.SentencePiece,
        org="Shanghai AI Lab",
    ),
    TokenizerConfig(
        "internlm/internlm2-math-7b",
        impl=TokenizerImpl.SentencePiece,
        org="Shanghai AI Lab",
    ),
    TokenizerConfig(
        "internlm/internlm-xcomposer-7b",
        impl=TokenizerImpl.SentencePiece,
        org="Shanghai AI Lab",
    ),
    TokenizerConfig("tiiuae/falcon-7b", impl=TokenizerImpl.SentencePiece, org="TII"),
    TokenizerConfig("tiiuae/falcon-180b", impl=TokenizerImpl.SentencePiece, org="TII"),
    TokenizerConfig(
        "Skywork/Skywork-13B-base", impl=TokenizerImpl.SentencePiece, org="Kunlun"
    ),
    TokenizerConfig(
        "Skywork/Skywork-13B-Math", impl=TokenizerImpl.SentencePiece, org="Kunlun"
    ),  # 文件：tokenizer.model
    TokenizerConfig(
        "FacebookAI/xlm-roberta-base", impl=TokenizerImpl.SentencePiece, org="Facebook"
    ),
    # 这个的tokenizer.json 为什么没有merges? vocab里为什么有概率值？
    # "goat",
    # ##### glm系列
    # "glm_chinese",),
    TokenizerConfig(
        "THUDM/chatglm-6b",
        impl=TokenizerImpl.SentencePiece,
        org="Tsinghua",
        meta=f"num_image_tokens: {12}; num_image_tokens: {34} ",
        init_kwargs={"revision": "refs/pr/100"},
    ),
    TokenizerConfig(
        "THUDM/chatglm2-6b",
        impl=TokenizerImpl.SentencePiece,
        org="Tsinghua",
    ),
    TokenizerConfig(
        "THUDM/chatglm3-6b",
        impl=TokenizerImpl.SentencePiece,
        org="Tsinghua",
    ),
    TokenizerConfig(
        "thu-coai/CharacterGLM-6B",
        impl=TokenizerImpl.SentencePiece,
        org="Tsinghua",
    ),
    # tiktoken 系列
    TokenizerConfig(
        "openai/text-davinci-003",
        impl=TokenizerImpl.TikToken,
        org="OpenAI",
        link="https://github.com/openai/tiktoken",
    ),
    #
    TokenizerConfig(
        "openai/code-davinci-002",
        impl=TokenizerImpl.TikToken,
        org="OpenAI",
        link="https://github.com/openai/tiktoken",
    ),
    TokenizerConfig(
        "openai/gpt-3.5-turbo",
        impl=TokenizerImpl.TikToken,
        org="OpenAI",
        link="https://github.com/openai/tiktoken",
        desc="tiktoken is a fast BPE tokeniser for use with OpenAI's models. There are 16 tokens KeyError",
    ),
    TokenizerConfig(
        "openai/gpt-4",
        impl=TokenizerImpl.TikToken,
        org="OpenAI",
        link="https://github.com/openai/tiktoken",
    ),
    TokenizerConfig(
        "openai/gpt-4o",
        impl=TokenizerImpl.TikToken,
        org="OpenAI",
        link="https://github.com/openai/tiktoken",
    ),
    TokenizerConfig(
        "Qwen/Qwen-7B-Chat",
        name_display="Qwen/Qwen",
        impl=TokenizerImpl.TikToken,
        org="Alibaba",
        init_kwargs={"revision": "refs/pr/56"},
        meta="在gpt4词典基础上，删除了100个多数字token，增加10000中文词token；并优化了special_token的分词",
    ),
    # https://huggingface.co/Qwen/Qwen-7B-Chat#%E6%A8%A1%E5%9E%8B%E7%BB%86%E8%8A%82%EF%BC%88model%EF%BC%89
    #  该词表在GPT-4使用的BPE词表cl100k_base基础上，对中文、多语言进行了优化，在对中、英、代码数据的高效编解码的基础上，
    #  对部分多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强。 词表对数字按单个数字位切分。
    # TokenizerConfig("Qwen/Qwen-72B-Chat", impl=TokenizerImpl.TikToken),
    # 未分类
    # ("amber", ""),
    TokenizerConfig("LLM360/CrystalCoder", org="MBZUAI"),
    TokenizerConfig("apple/DCLM-7B", org="Apple"),
    TokenizerConfig("mistralai/Mistral-7B-v0.1", org="Mistral"),
    TokenizerConfig("mistralai/Mixtral-8x7B-v0.1", org="Mistral"),
    TokenizerConfig("mistralai/Mistral-Large-Instruct-2407", org="Mistral"),
    TokenizerConfig("mistralai/Mistral-Nemo-Instruct-2407", org="Mistral"),
    TokenizerConfig("paust/pko-t5-large", org="PAUST"),
    TokenizerConfig("01-ai/Yi-6B", org="Yi"),
    TokenizerConfig("01-ai/Yi-34B", org="Yi"),
    TokenizerConfig("01-ai/Yi-VL-34B", org="Yi"),
    TokenizerConfig("01-ai/Yi-1.5-34B", org="Yi"),
    TokenizerConfig("OrionStarAI/Orion-14B-Chat", org="OrionStar"),
    TokenizerConfig("microsoft/phi-1", org="Microsoft"),
    TokenizerConfig("microsoft/phi-2", org="Microsoft"),
    TokenizerConfig(
        "microsoft/Phi-3-mini-4k-instruct", org="Microsoft", meta="即llama vocab"
    ),
    TokenizerConfig("Upstage/SOLAR-10.7B-v1.0", org="-"),
    TokenizerConfig("google/mobilebert-uncased", org="Google"),
    # ("google/mobilenet_v2_1.0_224",),  # error
    TokenizerConfig("google/switch-c-2048", org="Google"),
    TokenizerConfig("google/byt5-small", org="Google"),
    TokenizerConfig("google/mt5-large", org="Google"),
    TokenizerConfig("WizardLM/WizardCoder-Python-7B-V1.0", org="Microsoft"),
    TokenizerConfig("WizardLM/WizardCoder-15B-V1.0", org="Microsoft"),
    TokenizerConfig("WizardLM/WizardLM-7B-V1.0", org="Microsoft"),
    TokenizerConfig("WizardLM/WizardMath-70B-V1.0", org="Microsoft"),
    TokenizerConfig("TigerResearch/tigerbot-70b-chat-v4-4k", org="Tigerobo"),
    TokenizerConfig("TigerResearch/tigerbot-13b-chat-v2", org="Tigerobo"),
    TokenizerConfig("deepseek-ai/deepseek-coder-33b-instruct", org="DeepSeek"),
    TokenizerConfig("deepseek-ai/deepseek-llm-7b-base", org="DeepSeek"),
    TokenizerConfig("deepseek-ai/DeepSeek-V2", org="DeepSeek"),
    TokenizerConfig("deepseek-ai/DeepSeek-V3", org="DeepSeek"),
    TokenizerConfig(
        "deepseek-ai/DeepSeek-R1", org="DeepSeek"
    ),  # 在llama3的词典上，增加了一些中文token，删掉了一部分token
    TokenizerConfig("deepseek-ai/DeepSeek-R1-Zero", org="DeepSeek"),
    TokenizerConfig("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", org="DeepSeek"),
    TokenizerConfig("google/gemma-7b", org="Google"),
    TokenizerConfig("google/gemma-2-9b", org="Google"),
    TokenizerConfig("allenai/OLMo-7B-hf", org="Allen AI"),
    TokenizerConfig("HuggingFaceH4/zephyr-7b-beta", org="HuggingFace"),
    TokenizerConfig("ai21labs/Jamba-v0.1", org="AI21"),
    TokenizerConfig("databricks/dbrx-instruct", org="Databricks"),
    TokenizerConfig("MiniMaxAI/MiniMax-Text-01", org="MiniMax"),
    # TokenizerConfig("nvidia/Nemotron-4-340B-Instruct", org="Nvidia"),
    # ("claude",),
    # https://github.com/Duxiaoman-DI/XuanYuan
    # https://huggingface.co/apple/OpenELM-3B-Instruct  https://huggingface.co/apple/OpenELM-3B
]

assert len(set([config.name_display for config in _all_tokenizer_config])) == len(
    _all_tokenizer_config
)
assert len(set([config.name_or_path for config in _all_tokenizer_config])) == len(
    _all_tokenizer_config
)
assert len(
    set([config.name_or_path.split("/")[-1] for config in _all_tokenizer_config])
) == len(_all_tokenizer_config)


class TokenizerFactory:
    def __init__(self):
        # self.all_tokenizer_configs = sorted(_all_tokenizer_config, key=lambda k: k.name_or_path)
        self.all_tokenizer_configs = sorted(
            _all_tokenizer_config, key=lambda k: k.name_display
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

    def get_tokenizer_config(self, tokenizer_name: str) -> TokenizerConfig:
        for name_to_config in self.name_to_config_list:
            if tokenizer_name in name_to_config:
                return name_to_config[tokenizer_name]
        return None

    def get_tokenizer(self, tokenizer_name: str):
        """
        :param tokenizer_name:
        :return:
        """
        tokenizer_config = self.get_tokenizer_config(tokenizer_name)

        # 1. load from cache
        if tokenizer_config in self.tokenizer_cache:
            return self.tokenizer_cache[tokenizer_config]

        # 2. load tokenizer
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

    def add_config(
        self,
    ):
        pass

    def add_tokenizer(self, tokenizer_name):
        pass


tokenizer_factory = TokenizerFactory()


def add_tokenizer(tokenizer_name: str):
    """
    :param tokenizer_name:
    :return:
    """
    if tokenizer_name in []:
        logger.info(f"{tokenizer_name} already exits")
    else:
        # add to config
        tokenizer_config = TokenizerConfig(tokenizer_name, org="-")

        # add to tokenizer
        tokenizer = tokenizer_factory.load_tokenizer(tokenizer_config)

        # refresh cache

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True, **tokenizer_config.init_kwargs
            )
            tokenizer_factory.all_tokenizer_configs.append(
                "",
            )
            tokenizer_factory

        except Exception as e:
            logger.error(e)

    pass


# class TokenizerType(Enum):
#
#     # BERTTokenizer
#     # 依赖一个txt文件
#
#
#     # https://github.com/EleutherAI/gpt-neox/blob/v2.0/megatron/tokenizer/tokenizer.py#L231
#     # 依赖一个json文件，Tokenizer.from_file(vocab_file)
#     # 案例：gpt-neox-20B
#     HFTokenizer = auto()
#
#     # 依赖: model_file, sentencepiece.SentencePieceProcessor(model_file)
#     # 案例：
#     SentencePieceTokenizer = auto()
#
#
#     # 依赖: 3个json文件：vocab.json, merges.txt, special_tokens.txt
#     # 源码:
#     #   - https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/tokenizer/gpt2_tokenization.py#L92
#     # Byte-level BPE
#     GPT2BPETokenizer = auto()


if __name__ == "__main__":
    for tokenizer_config in tokenizer_factory.all_tokenizer_configs:
        if True:
            # if "t5" in tokenizer_config.name_or_path:
            tokenizer1 = tokenizer_factory.get_tokenizer(tokenizer_config.name_or_path)
            tokenizer2 = tokenizer_factory.get_tokenizer(tokenizer_config.name_display)
            tokenizer3 = tokenizer_factory.get_tokenizer(
                tokenizer_config.name_display.split("/")[-1]
            )
            assert tokenizer1 == tokenizer2 == tokenizer3
            print(tokenizer_config.name_or_path, len(tokenizer1))
