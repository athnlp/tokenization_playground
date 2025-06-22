import json

from vocab import TokenizerImpl, all_tokenizer_config, load_tokenizer

text = (
    "hello; Замглавы управления развития; 특히 주소 15~17번 홀에선 3연속;"
    " 確実に春が近づいてること;  a közoktatással? _ Belföld;"
    " pumë, i vjetër, vjeç; ئەردوغان ۋە قىرغىزىستان ;"
    " निम्न में से कौन सा हारडवेयर; ተለዋዋጭ የግድግዳ ; Дзейныя асобы:;"
    " « અમરેલીનાં મહિલા વિકાસ; 🦙❤❥웃유♋☮✊;"
    "װיקיװערטערבוך "
)
whitespace = "\t   \n\n\r  "
bytes = b"\x00\x01\x02\x03\x04".decode("utf-8")

text += whitespace


def get_unk(tokenizer_config):
    tokenizer = load_tokenizer(tokenizer_config)
    if hasattr(tokenizer, "unk_token"):
        return f"{tokenizer.unk_token}, {tokenizer.unk_token_id}"
    else:
        return "unk_token not found"


# def infer_tokenizer_impl(tokenizer_config):
def infer_tokenizer_type(tokenizer_config):
    tokenizer = load_tokenizer(tokenizer_config)
    if tokenizer_config.impl == TokenizerImpl.TikToken:
        return "tiktoken"
    if hasattr(tokenizer, "backend_tokenizer"):
        return str(
            type(tokenizer.backend_tokenizer.model)
        )  # type(tokenizer._tokenizer.model))
    # orion: sp_model.Load(vocab_file)，继承 PreTrainedTokenizer
    elif hasattr(tokenizer, "sp_model"):  # 基于 sentencepiece 包
        # for i in range(tokenizer.sp_model.piece_size()):
        #     if tokenizer.sp_model.is_byte(i):
        #         print("")
        return f"sp_model, byte_num: {sum([tokenizer.sp_model.is_byte(i) for i in range(tokenizer.sp_model.piece_size())])}"

    # sp.Load(model_path)  ，并且包括image_tokenizer
    elif "glm-" in tokenizer_config.name_or_path:
        return f"byte_num: {sum([tokenizer.sp_tokenizer.text_tokenizer.sp.is_byte(i) for i in range(tokenizer.sp_tokenizer.text_tokenizer.sp.piece_size())])}"
    # sp.Load(model_path)  ，没有image_tokenizer
    elif (
        "glm2-" in tokenizer_config.name_or_path
        or "glm3-" in tokenizer_config.name_or_path
        or "CharacterGLM-6B" in tokenizer_config.name_or_path
    ):
        return f"byte_num: {sum([tokenizer.tokenizer.sp_model.is_byte(i) for i in range(tokenizer.tokenizer.sp_model.piece_size())])}"
    elif (
        "abeja/gpt-neox-japanese-2.7b" == tokenizer_config.name_or_path
    ):  # 支持 byte-level，解决oov问题
        return "japanese-bpe: https://github.com/tanreinama/Japanese-BPEEncoder_V2"
    # bert-base-japanese： 特殊的地方在于 "word_tokenizer_type": "mecab"，见 https://huggingface.co/tohoku-nlp/bert-base-japanese/blob/main/tokenizer_config.json
    elif "bert-base-japanese" in tokenizer_config.name_or_path:
        return (
            "wordpiece.MecabTokenizer, 支持byte-level https://taku910.github.io/mecab/"
        )
    elif "moss" in tokenizer_config.name_or_path:
        return "应该是 sentencepiece.byte_bpe,待确认"
    elif "byt5" in tokenizer_config.name_or_path:
        return "未知，待定"
    else:
        print("catch", tokenizer_config.name_or_path)
        raise "error"


def test_lossless(tokenizer_config):
    """
    xlm-roberta-base 为什么oov这么少？是因为有 byte吗？
    :param tokenizer_config:
    :return:
    """
    tokenizer = load_tokenizer(tokenizer_config)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    decoding = tokenizer.decode(encoding)

    if text in decoding:
        # print(tokenizer_config.name, tokenizer_config.impl, "lossless: true")
        pass
    else:
        unk_count = sum(
            [1 for token_id in encoding if token_id == tokenizer.unk_token_id]
        )
        oov_tokens = []
        # if tokenizer_config.impl == TokenizerImpl.SentencePiece:
        #     print(sum([tokenizer.is_byte(i) for i in range(tokenizer.piece_size())]))

        print("#######" * 5)
        print(
            f"{tokenizer_config.name_or_path}, {infer_tokenizer_type(tokenizer_config)}\n"
            f"lossless: false; unk_token: {get_unk(tokenizer_config)},"
            f" unk_ratio: {unk_count/len(encoding):.4f}; oov: []"
        )
        for i in range(len(text)):
            if text[i] != decoding[i]:
                # print(f"text[{i}]     = {str(bytes(text[i:], 'utf-8'))}\n"
                #       f"decoding[{i}] = {str(bytes(decoding[i:], 'utf-8'))}")
                print(
                    f"text[{i}]     = {json.dumps(text[i:], ensure_ascii=False)}, \n"
                    f"decoding[{i}] = {json.dumps(decoding[i:], ensure_ascii=False)}"
                )

                break


for config in all_tokenizer_config:
    # if "xlm-roberta-base" in config.name:
    # if "xlm-roberta-base" in config.name:
    # if "chatglm3-6b" in config.name:
    # if "bert-base-japanese" in config.name:
    # if "moss" in config.name:
    # if "byt5" in config.name:
    if "baichuan" in config.name_or_path:
        # if "CharacterGLM-6B" in config.name:
        # if "fastchat-t5" in config.name:  # 报错 pyo3_runtime.PanicException: AddedVocabulary bad split
        # if True:
        # test_unk(config)
        test_lossless(config)
