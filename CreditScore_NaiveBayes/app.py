

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        kpr_aktif_TIDAK = float(request.form['kpr_aktif_TIDAK'])
        kpr_aktif_YA = float(request.form['kpr_aktif_YA'])
        rata1 = float(request.form['rata1'])
        rata2 = float(request.form['rata2'])
        rata3 = float(request.form['rata3'])
        rata4 = float(request.form['rata4'])
        rata5 = float(request.form['rata5'])
        pendapatan_setahun_juta = float(request.form['pendapatan_setahun_juta'])
        durasi_pinjaman_bulan = float(request.form['durasi_pinjaman_bulan'])
        jumlah_tanggungan = float(request.form['jumlah_tanggungan'])


        val = np.array([kpr_aktif_TIDAK, kpr_aktif_YA, rata1,rata2, rata3, rata4, rata5, pendapatan_setahun_juta, durasi_pinjaman_bulan, jumlah_tanggungan])

        final_features = [np.array(val)]
        model_path = os.path.join('model_credit_NB.pkl')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)

        return render_template('index.html', result=res)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)