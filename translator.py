from transformers import pipeline

class Translator:

    def __init__(self):
        # Load the translation pipeline
        self.translator = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en")

    def translate(self, german_text):
        
        # Split the text into sentences
        sentences = german_text.split('. ')

        # Translate each sentence individually and handle punctuation
        translations = [self.translator(sentence)[0]['translation_text'] for sentence in sentences]

        # Combine the translated sentences with proper punctuation
        translated_text = '. '.join(translations)

        # Output the translated text
        return translated_text

if __name__ == "__main__":

    german_example = "Dieser dämliche Idiot ist aus dem Tunnel abgehauen. Ich kann weitergraben!"
    translator = Translator()
    translated_text = translator.translate(german_example)
    print(translated_text)