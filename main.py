import streamlit as st
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
import contractions
import numpy as np
import sklearn
import pickle
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def prediction(text, n):
    if TextBlob(text).sentiment.polarity >= 0:
        return "Cet avis n'apparait pas comme negatif"
    else:
        # tokenize
        text = contractions.fix(text).lower()
        tokenizer = RegexpTokenizer(r"\w+")
        text = tokenizer.tokenize(text)
        # print("Tokens:",text)

        # remove stops words
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        # print("Stop words:",text)

        # lemmatize
        lemmatizer = WordNetLemmatizer()
        token_tag = nltk.pos_tag(text)
        lemm_text = []
        for word, tag in token_tag:
            if tag.startswith('J'):
                lemm_text.append(lemmatizer.lemmatize(word, 'a'))
            elif tag.startswith('V'):
                lemm_text.append(lemmatizer.lemmatize(word, 'v'))
            elif tag.startswith('N'):
                lemm_text.append(lemmatizer.lemmatize(word, 'n'))
            elif tag == 'PRP':
                lemm_text.append(word)
            elif tag.startswith('R'):
                lemm_text.append(lemmatizer.lemmatize(word, 'r'))
            else:
                lemm_text.append(lemmatizer.lemmatize(word))
        # print("Lemmatized:",lemm_text)

        # join
        clean_text = ' '.join(lemm_text)
        # print("Join:", clean_text )

        # vectorize
        with open('vectoriseur', 'rb') as file:
            vectorizer = pickle.load(file)
        X = vectorizer.transform([clean_text])
        # print("Vectorised :", X)

        # predict
        with open('model', 'rb') as file:
            model = pickle.load(file)
        W = model.transform(X)
        # print("Matrice topic:", W.argmax())

        # topic
        topic = {'topic1': 'ACCUEIL ET SERVICE', 'topic2': 'NOURRITURE ASIATQUE MAUVAISE ',
                 'topic3': "TEMPS D'ATTENTE ET PIZZA FROIDE OU TROP CUITE",
                 'topic4': 'MAUVAISE EXPERIENCE AVEC LE PESONNEL ',
                 'topic5': 'PROBLEME BURGER (ERREUR COMMANDE, PRIX ELEVE, ETC)', 'topic6': "TEMPS D'ATTENTE TROP LONG",
                 'topic7': 'PROBLEME QUALITE BURGER', 'topic8': 'TRES MAUVAIS SERVICE CLIENT',
                 'topic9': 'PROBLEME DE POULET', 'topic10': 'BAR MAUVAIS EXPEIENCE AVEC LE PERSONNEL',
                 'topic11': 'CLIENT DECU, NE REVIENDRA PAS', 'topic12': 'NOURRITURE JAPONNAISE DECEVANTE',
                 'topic13': 'MAUVAIS SANDWICH', 'topic14': 'EXPERIENCE MEDIOCRE, PRIX LEGEREMENT TROP ELEVES',
                 'topic15': 'PROBLEME DANS LA COMMANDE'}

        # print("Topic: ", list(topic.values())[W.argmax()])
        topics = []
        results = np.argsort(W)
        print(results)

        for index in results[0][-n:]:
            print(type(index), index)
            topics.append(list(topic.values())[int(index)])

        # result_sorted = np.argsort(results)
        # result_sorted = topic.values().sort()
        # topics = result_sorted[:n]
        topics.reverse()
        return topics
        # return list(topic.values())[W.argmax()]


st.title('Topic Review identifier')
txt = st.text_area('Insert review here', '')
n = st.slider('Number of topics displayed', 1, 15)
click = st.button('Validate')

if click:
    st.write(prediction(txt, n))
