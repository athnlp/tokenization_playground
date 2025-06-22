import gradio as gr


def get_lang(request: gr.Request):
    """
    'accept-language', b'zh,en;q=0.9,zh-CN;q=0.8')
    """
    accept_language = None
    langs = []
    try:
        accept_language = request.headers["Accept-Language"]
        for lang in accept_language.split(",")[:5]:
            lang = lang.lower()
            if lang.startswith("en"):
                langs.append("en")
            elif lang.startswith("es"):
                langs.append("es")
            elif lang.startswith("zh"):
                langs.append("zh")
            elif lang.startswith("fr"):
                langs.append("fr")
            elif lang.startswith("de"):
                langs.append("de")
    except Exception as e:
        print(e)
    return accept_language, langs
