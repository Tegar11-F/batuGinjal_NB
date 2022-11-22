

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        gravity = float(request.form['gravity'])
        ph = float(request.form['ph'])
        osmo = float(request.form['osmo'])
        cond = float(request.form['cond'])
        urea = float(request.form['urea'])
        calc = float(request.form['calc'])


        val = np.array([gravity, ph, osmo,cond, urea,calc])

        final_features = [np.array(val)]
        model_path = os.path.join('model_batuginjal_NB.pkl')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)

        return render_template('index.html', result=res)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)