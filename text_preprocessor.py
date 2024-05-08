from tqdm import tqdm
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import pickle

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(pickle.load(open('stopwords.pkl','rb')))

    def remove_punctuation(self,text):
        punctuationfree="".join([i if i not in string.punctuation else ' ' for i in text ])
        return punctuationfree


    # Converting the words having apostophe into their root form
    def decontracted(self,phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"there\'s", "there is", phrase)
        phrase = re.sub(r"it\'s", "it is", phrase)
        phrase = re.sub(r"he\'s", "he is", phrase)
        phrase = re.sub(r"she\'s", "she is", phrase)
        phrase = re.sub(r"how\'s", "how is", phrase)
        phrase = re.sub(r"let\'s", "let is", phrase)
        phrase = re.sub(r"so\'s", "so is", phrase)
        phrase = re.sub(r"what\'s", "what is", phrase)
        phrase = re.sub(r"when\'s", "when is", phrase)
        phrase = re.sub(r"where\'s", "where is", phrase)
        phrase = re.sub(r"why\'s", "why is", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def lemmatize_text(self,preprocessed_text):
        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            # print(tag)
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        # 1. Init Lemmatizer
        lemmatizer = WordNetLemmatizer()
       # 2. Lemmatize a Sentence with the appropriate POS tag
        lemmatized_text=[lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(preprocessed_text)]
        # The final processed set of words for each iteration will be stored in 'text_final'
        text_final=(" ".join( lemmatized_text ))
        return text_final

    # Combined Data Cleaning function
    def preprocessing(self,sentences):
        # Lowercasing
        sentences=[x.lower() for x in sentences]
        # Handling apostophe
        sentences=[self.decontracted(x) for x in sentences]
        # Handling numerics
        sentences=[re.sub(r"\d", "",s) for s in sentences]
        # Removing punctuations
        sentences=[re.sub(r"\n", " ",s) for s in sentences]
        sentences=[self.remove_punctuation(f) for f in sentences]
        # Removing unicodes from strings 
        sentence=[s.encode('ascii', 'ignore').decode() for s in sentences]
        #  Lemmatization
        ans=[]
        for x in tqdm(sentences,total=len(sentences)):
            ans.append(self.lemmatize_text(x))
        sentences=ans.copy()   
        # Stop word removal
        sentences=[' '.join([word for word in x.split() if word not in self.stop_words]) for x in sentences]
        return sentences