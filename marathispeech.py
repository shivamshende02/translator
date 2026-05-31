from gtts import gTTS

# Marathi text
text = "हरीला त्याच्या गांडवर मारले पाहिजे."

# Convert text to speech
tts = gTTS(text=text, lang='mr', slow=True)

# Save as MP3
tts.save("output.mp3")

print("MP3 file saved as output.mp3")
