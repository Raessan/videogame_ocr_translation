from ocr_german import OCR
from translator import Translator
from params import *
import requests
from flask import Flask, request, render_template, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np


ocr_german = OCR(region, threshold_binary, min_zeros_between_lines, min_nonzeros_line, min_zeros_between_characters, min_zeros_space, folder_chars, chars_dict, size_characters, save_cnn_folder, save_cnn_file)
translator = Translator()
app = Flask(__name__)

def get_german_text_and_image():

    ocr_german.get_chars_from_image()
    _, _, _, img_characters_segmented = ocr_german.segment_characters_and_plot()

    # Convert numpy array to PIL image
    if isinstance(img_characters_segmented, np.ndarray):
        img_characters_segmented = Image.fromarray(img_characters_segmented)

    buffered = BytesIO()
    img_characters_segmented.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return ocr_german.text, img_str

def reset_german_text():
    ocr_german.reset_text()
    return "", "", None

def translate_german_text():
    return translator.translate(ocr_german.text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_text', methods=['POST'])
def get_text():
    german_text, img_str = get_german_text_and_image()
    return jsonify({'german_text': german_text, 'image': img_str})

@app.route('/translate_text', methods=['POST'])
def translate_text():
    english_text = translator.translate(ocr_german.text.lower())
    return jsonify({'english_text': english_text})

@app.route('/reset_text', methods=['POST'])
def reset_text():
    reset_german_text()
    return jsonify({'german_text': ''})

def get_existing_words():
    """ Helper function to retrieve existing words from the save file. """
    existing_words = set()
    try:
        with open(save_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.split(':')[0].strip()
                existing_words.add(word)
    except FileNotFoundError:
        pass  # File does not exist initially, or any other error handling needed
    return existing_words

@app.route('/save_word', methods=['POST'])
def save_word():
    word = request.form.get('word')
    translated_word = translator.translate(word.lower())

    # Check if the word is already saved
    existing_words = get_existing_words()
    if word in existing_words:
        return jsonify({'message': 'Word already exists in the file'})

    # Write the word to the file
    with open(save_file_path, 'a', encoding='utf-8') as file:
        file.write(word + ': ' + translated_word + '\n')

    return jsonify({'message': 'Word saved successfully'})

if __name__ == '__main__':
    app.run(debug=True)