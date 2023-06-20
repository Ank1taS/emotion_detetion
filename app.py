import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask,request,app,jsonify, url_for, render_template
# from clean_data import get_clean_text
import clean_data


app = Flask(__name__)

# load data-cleaning steps,data pre-processing steps and  model
text_clean_function = pickle.load(open('clean_text.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidfconverter = pickle.load(open('transform.pkl', 'rb'))
pickled_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))


# @app.route('/predict_api', method=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
    


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    txt = request.form['sentence']
    # txt_list = list(txt)

    # print(txt)

    txt = text_clean_function(txt)
    fm = vectorizer.transform([txt])
    tfidf = tfidfconverter.fit_transform(fm).toarray()
    emotion = pickled_model.predict(tfidf)

    # print(emotion)
    # ouytput 

    return render_template('home.html', predicted_emotion='{}'.format(emotion[0].upper()))


if __name__ == "__main__":
    app.run(debug=True)

