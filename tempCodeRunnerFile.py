from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model_hasil = joblib.load('knn.pkl')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    data_prediksi = request.files['prediksi']
    data_prediksi_path = './prediksi/' + data_prediksi.filename
    data_prediksi.save(data_prediksi_path)

    csv_file = request.files.get('prediksi')
    X_test = pd.read_csv(csv_file)
    X_test['prediksi'] = model_hasil.predict(X_test)
    return render_template('index.html', prediction = X_test.to_html())



if __name__ == '__main__':
    app.run(port=3000, debug=True)