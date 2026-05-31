from googletrans import Translator

translator = Translator()
text = "The law may seem complex, and opaque, but understanding some of its key terminology can make it more accessible and less intimidating."
translated_text = translator.translate(text, src="en", dest="mr")
print(translated_text.text)
