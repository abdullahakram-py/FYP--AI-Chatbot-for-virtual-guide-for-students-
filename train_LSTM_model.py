import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Bidirectional, LSTM
from keras.optimizers import Adam
import random
from keras.preprocessing.sequence import pad_sequences


words = []
classes = []
documents = []
ignore_words = ['?', '!', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do',
                'does', 'did', 'can', 'could', 'will', 'would', 'should', 'shall', 'may', 'might', 'must',
                'get', 'got', 'any', 'some', 'all', 'this', 'that', 'these', 'those', 'it', 'its', 'they',
                'them', 'their', 'theirs', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him',
                'his', 'she', 'her', 'hers', 'i', 'me', 'my', 'mine', 'and', 'or', 'but', 'nor', 'so', 'for',
                'yet', 'after', 'before', 'with', 'without', 'under', 'over', 'at', 'on', 'in', 'out', 'off',
                'up', 'down', 'to', 'from', 'by', 'about', 'as', 'of', 'between', 'into', 'through', 'during',
                'if', 'else', 'then', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose',
                'am', 'is', 'are', 'been', 'being', 'have', 'has', 'had', 'will', 'would', 'shall', 'should',
                'can', 'could', 'may', 'might', 'must']

data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# create our training data
training = []
output_empty = [0] * len(classes)
max_len = max([len(doc[0]) for doc in documents])  # Find the maximum sequence length

# create an embedding matrix
vocab_size = len(words)
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_x = pad_sequences(train_x, maxlen=max_len, padding='post', truncating='post')
train_y = list(training[:, 1])
print("Training data created")

# Create model - Bidirectional LSTM
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=32, verbose=1)
model.save('chatbot_model.h5', hist)

print("Model created")