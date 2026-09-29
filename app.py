from flask import Flask, render_template, request
from chatgui import chatbot_response
from spellchecker import SpellChecker
from chatgui import check_random_text
import string

app = Flask(__name__)

def spell_check_and_correct(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = {}

    # Check each word for spelling errors
    for word in words:
        # Check if the word is misspelled
        if not spell.correction(word) == word:
            # If misspelled, store the corrected word
            corrected_words[word] = spell.correction(word)

    return " ".join(corrected_words.get(word, word) for word in text.split())

def convert_abbreviations(user_input):
    abbreviations = {
        "CLO": "Course Learning Outcome",
        "CLOS": "Course Learning Outcomes",
        "PLO": "Program Learning Outcome",
        "PLOS": "Program Learning Outcomes",
        "EPC": "English Proficiency Certificate",
        "DMC": "Detailed Mark Sheet"
    }

    # Split the input string into words
    words = str(user_input).split()

    # Iterate over the words and replace abbreviations with their full forms
    converted_words = []
    for word in words:
        # Remove any punctuation marks from the word
        cleaned_word = ''.join(char for char in word if char not in string.punctuation)

        if cleaned_word.upper() in abbreviations:
            converted_words.append(abbreviations[cleaned_word.upper()])
        else:
            converted_words.append(word)
    # Join the converted words back into a string
    converted_text = " ".join(converted_words)
    print("User Input: ", user_input)
    print("Converted Text: ", converted_text)

    return converted_text


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    
    if check_random_text(msg):
        output = "I'm sorry, I could not understand."
    else:
        msg = convert_abbreviations(msg)
        corrected_msg = spell_check_and_correct(msg)
        output = chatbot_response(corrected_msg)
        
    return output

if __name__ == '__main__':
    app.run()
