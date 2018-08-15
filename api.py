from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from core import main
from os.path import join


app = Flask(__name__)
CORS(app)
app.config['FOLDER_POSITIVE'] = 'static/img/positive/'
app.config['FOLDER_NEGATIVE'] = 'static/img/negative/'


@app.route('/', methods=['GET'])
def api_root():
    msg = 'Image-Classification-Toolkit permite realizar una clasificación dual a partir de dos conjuntos de imágenes (positivas y negativas) con los algoritmos SVM-KNN-BPNN-CNN y Transfer Learning .'
    return render_template('index.html', msg=msg)


@app.route('/file-upload/<type>', methods=['POST'])
def file_upload_positive(type):
    new_file = request.files.get('file', None)
    file_name = new_file.filename.split('.')[0]
    if new_file is not None:
        if type == 'positive':
            new_file.save(join(app.config['FOLDER_POSITIVE'], file_name))
            response = jsonify({'success': True, 'file_name': file_name, 'type': type})
            response.status_code = 200
            return response
        elif type == 'negative':
            new_file.save(join(app.config['FOLDER_NEGATIVE'], file_name))
            response = jsonify({'success': True, 'file_name': file_name, 'type': type})
            response.status_code = 200
        else:
            response = jsonify({'success': False, 'message': type + ' no corresponde a ningun tipo valido (positive-negative) '})
            response.status_code = 400
            pass
        return response
    response = jsonify({'success': False, 'message': 'Error al leer el archivo'})
    response.status_code = 400
    return response


@app.route('/next-form', methods=['GET'])
def next_form():
    return 'hehe'


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
