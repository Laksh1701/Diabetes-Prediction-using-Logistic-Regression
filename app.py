import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,send_from_directory
import pickle
from os import path
from werkzeug.utils import secure_filename
app = Flask(__name__)


MODEL= 'my_model.pkl'
model = pickle.load(open(f'./models/{MODEL}','rb'))
SCALAR= 'transformer.pkl'
sc = pickle.load(open(f'./models/{SCALAR}','rb'))
UPLOAD_FOLDER = 'uploads'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features))

    if prediction == 1:
        pred = "You have Diabetes, Please consult a Doctor !"
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
