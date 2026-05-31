from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = "aryaumesh/english-to-marathi"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def translate_to_marathi(text):
    tokenizer.src_lang = "en_XX"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Debug: Check available language codes
    print("Available language codes:", tokenizer.lang_code_to_id)

    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("mr_IN"))

    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

english_text = "Hello, how are you?"
print("Marathi Translation:", translate_to_marathi(english_text))