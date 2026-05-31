from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 

def translate_to_marathi(text):
    
    model_name = "aryaumesh/english-to-marathi"

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Generate translation
    translated_tokens = model.generate(**inputs)

    # Decode translation
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

translate_to_marathi("Hello")
