from flask import Flask,render_template,url_for,request, jsonify
import pandas as pd 
import numpy as np
import pickle
import re
import nltk 
import tensorflow as tf
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import json

# load the model from disk
#global graph
#graph = tf.get_default_graph()

path = 'C:/Users/Nik/Desktop/cdpl1/model2_gv_deepl.h5'
model = tf.keras.models.load_model(path)
#graph = tf.get_default_graph()

filename = 'C:/Users/Nik/Desktop/cdpl1/tokenizer.pkl'
tokenizer = pickle.load(open(filename, 'rb'))
#cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)


def preprocess(text):
    
    """performs common expansion of english words, preforms preprocessing"""
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"won\'t", "will not", text)   # decontracting the words
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = re.sub(r'\w+:\s?','',text)                                            ## removing anyword:
    text = re.sub('[([].*?[\)]', '', text)                                       ## removing sq bracket and its content
    text = re.sub('[<[].*?[\>]', '', text)                                       ## removing <> and its content
    text = re.sub('[{[].*?[\}]', '', text)                                       ## removing {} and its content
    
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])       ## lemmatizing the word

    text = re.sub(r'\W', ' ', str(text))                                         # Remove all the special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)                                  # remove all single characters 
    text = re.sub(r"[^A-Za-z0-9]", " ", text)                                    # replace all the words except "A-Za-z0-9" with space  
    text = re.sub(r'[^\w\s]','',text)
    text = ' '.join(e for e in text.split() if e.lower() not in set(stopwords.words('english')) and len(e)>2)  
    # convert to lower and remove stopwords discard words whose len < 2
    
    text = re.sub("\s\s+" , " ", text)                                           ## remove extra white space  lst
    text = text.lower().strip()   

    return text

###################################################


#@app.route('/')
#def hello_world():
#    return 'Hello World!'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST']) 
#@app.route('/end_to_end_pipeline', methods=['POST'])

def predict():
  #path = 'C:/Users/Nik/Desktop/cdpl1/model2_gv_deepl.h5'
  result = []
  
  data = request.form.values()
  data = str(data)
  x = preprocess(data)
  sent_token = tokenizer.texts_to_sequences([x])

  sent_token_padd = pad_sequences(sent_token, maxlen=300, dtype='int32', padding='post', truncating='post')
  #model = tf.keras.models.load_model(path)
  pred = model.predict(sent_token_padd, batch_size=64)
  
  row, column = pred.shape
  prdt = np.zeros((row, column)) 
  for i in range(row):
    for j in range(column):
      if pred[i,j]>0.5:
        prdt[i,j] = 1
  
  if request.method == 'POST':
    
    for k in range(prdt.shape[0]):
      if prdt[k][0] == 1.0:
        result.append('commenting')
      if prdt[k][1] == 1.0:
        result.append('ogling')
      if prdt[k][2] == 1.0:
        result.append('groping')
      if np.sum(prdt) == 0.0:
        result.append('None')
        
    #return jsonify({'predict': result})
  return render_template('index.html',prediction_text='Possible action is {}'.format(result))
    

if __name__ == '__main__':
	app.run(debug=True)