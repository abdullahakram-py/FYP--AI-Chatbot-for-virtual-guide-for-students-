import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from better_profanity import profanity
from nltk.corpus import words as nltk_words


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                # if show_details:
                    # print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    try:    
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                # print(f"Tag: {tag}")
                # print("Result: ", result)
                break
    except:
        result = "Sorry, I could not understand"        
    finally:
        return result
    
def check_random_text(input_text):
    word_list = set(nltk_words.words())
    tokens = nltk.word_tokenize(input_text)
    
    if len(tokens) == 0:  # Check to avoid division by zero
        return False
    
    non_words = [word for word in tokens if word.lower() not in word_list]
    if len(non_words) / len(tokens) > 0.5:
        return True
    return False


def chatbot_response(msg):
    global lemmatizer
    global model
    global intents
    global words
    global classes

    lemmatizer = WordNetLemmatizer()
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json', encoding="utf-8").read())
    words = pickle.load(open('words.pkl','rb'))
    classes = pickle.load(open('classes.pkl','rb'))
    
    # Check for profanity in the user's input
    if profanity.contains_profanity(msg):
        return "Please refrain from using abusive language."
    
    # Check random text
    if check_random_text(msg):
        return "I'm sorry, I could not understand."
    
    # Check for proper context
    if len(msg) <4 and msg not in ["hi", "hey", "bye"]:
        return "Please provide more information."
        
    ints = predict_class(msg, model)
    
    res = getResponse(ints, intents)
    return res

def send(event=None):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#31154F", font=("Verdana", 12 ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


if __name__ == "__main__":
    
    #Creating GUI with tkinter
    import tkinter
    from tkinter import *

    base = Tk()
    base.title("Chatbot")
    base.geometry("800x650")
    base.resizable(width=FALSE, height=FALSE)

    #Create Chat window
    ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

    ChatLog.config(state=DISABLED)

    #Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set

    #Create Button to send message
    SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                        bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                        command= send )

    #Create the box to enter message
    EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
    EntryBox.bind("<Return>", send)

    #Place all components on the screen
    scrollbar.place(x=776,y=6, height=556)
    ChatLog.place(x=6,y=6, height=556, width=766)
    EntryBox.place(x=128, y=570, height=70, width=641)
    SendButton.place(x=6, y=570, height=70)

    base.mainloop()
