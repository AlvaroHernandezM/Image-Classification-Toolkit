from flask import Flask
from flask_cors import CORS
from core import main

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def api_root():
    return 'Image-Classification-Toolkit permite realizar una clasificación dual a partir de dos conjuntos de imágenes (positivas y negativas) con los algoritmos SVM-KNN-BPNN-CNN y Transfer Learning .'


@app.route('/classification_cnn', methods=['GET'])
def classification_cnn():
    return main.train('cnn', 770, 2, 2000)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
