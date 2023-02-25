# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 00:34:44 2022

@author: rupesh
"""

# -*- coding: utf-8 -*-

import emoji
import nltk
nltk.download('wordnet')
import pandas as pd
import warnings
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import re
#from rake_nltk import Rake
import pickle
import streamlit as st
#import numpy as np
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from selenium import webdriver
import time
#from webdrivermanager.chrome import ChromeDriverManager 
import turtle


 
pickle_in = open(r"C:\Users\lenovo\Desktop\telecc\Linear .pkl","rb")
data=pickle.load(pickle_in)

pickle_in = open(r"C:\Users\lenovo\Desktop\telecc\tfidf.pkl","rb")
tfidf=pickle.load(pickle_in)

st.header("Hotel Review Web Application")



input_text = st.text_area("Type your review here")
    
if st.button("Test Review"):
      
      wordnet=WordNetLemmatizer()
      sentence=re.sub('[^A-za-z0-9]',' ',input_text)
      sentence=sentence.lower()
      sentence=sentence.split(' ')
      sentence = [wordnet.lemmatize(word) for word in sentence if word not in (stopwords.words('english'))]
      sentence = ' '.join(sentence)
      
      pickle_in = open(r"C:\Users\lenovo\Desktop\telecc\Linear .pkl", 'rb') 
      data = pickle.load(pickle_in)
      pickle_in = open(r"C:\Users\lenovo\Desktop\telecc\tfidf.pkl", 'rb') 
      
      tfidf = pickle.load(pickle_in)
      transformed_input = tfidf.transform([sentence])
      
      if data.predict(transformed_input) == 0:
          st.write("Negative Review")
      elif data.predict(transformed_input) == 1:
          st.write("Positive Review")


   


#def main():
    #turtle.Screen().bgcolor("pink")
    
    



#st.snow()
#st.subheader("Enter the review for the prediction")
