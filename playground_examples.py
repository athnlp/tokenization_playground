default_user_input = """Replace this text in the input field to see how tokenization works."""
default_tokenizer_name_1 = "openai/gpt-4o"
default_tokenizer_name_2 = "Qwen/Qwen2.5-72B"


number_example = """127+677=804
127 + 677 = 804\n
1275+6773 = 8041
1275 + 6773 = 8048"""

code_example = """for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
"""

spelling_example = """How do you spell "accommodate"?
How many letters are in the word "accommodate"?
How many r's are in the word strawberry?"""

examples = {
    "number": {
        "text": number_example,
        "tokenizer_1": default_tokenizer_name_1,
        "tokenizer_2": default_tokenizer_name_2,
    },
    "code": {
        "text": code_example,
        "tokenizer_1": default_tokenizer_name_1,
        "tokenizer_2": default_tokenizer_name_2,
    },
    "spelling": {
        "text": spelling_example,
        "tokenizer_1": default_tokenizer_name_1,
        "tokenizer_2": default_tokenizer_name_2,
    },
}
