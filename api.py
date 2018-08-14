from flask import Flask
from flask_cors import CORS
from core import main

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def api_root():
    return 'Image-Classification-Toolkit permite realizar una clasificación dual a partir de dos conjuntos de imágenes (positivas y negativas) con los algoritmos SVM-KNN-BPNN-CNN y Transfer Learning .'


@app.route('/train_cnn/<steps_per_epoch>/<epochs>/<validation_steps>/<positive_class>/<negative_class>', methods=['GET'])
def train_cnn(steps_per_epoch, epochs, validation_steps, positive_class, negative_class):
    return main.train_cnn(int(steps_per_epoch), int(epochs), int(validation_steps), positive_class, negative_class)


@app.route('/classification_cnn', methods=['GET'])
def classification_cnn():
    return main.classification_cnn()


@app.route('/train_image_retraining')
def train_image_retraining():
    return main.train_image_retraining()


@app.route('/classification_image_retraining')
def classification_image_retraining():
    return main.classification_image_retraining()


@app.route('/train_svm_knn_bpnn/<number_neighbors>/<focus>/<hidden_layer_sizes>/<max_iter_bpnn>/<max_iter_svm>')
def train_svm_knn_bpnn(number_neighbors, focus, hidden_layer_sizes, max_iter_bpnn, max_iter_svm):
    return main.train_svm_knn_bpnn(int(number_neighbors), focus, int(hidden_layer_sizes), int(max_iter_bpnn), int(max_iter_svm))


@app.route('/classification_svm_knn_bpnn')
def classification_svm_knn_bpnn():
    return main.classification_svm_knn_bpnn()


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
