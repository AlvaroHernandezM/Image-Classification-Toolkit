#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from core import main
from os.path import join
import utils

app = Flask(__name__)
CORS(app)
app.config['FOLDER_POSITIVE'] = 'static/img/positive/'
app.config['FOLDER_NEGATIVE'] = 'static/img/negative/'
app.config['FOLDER_IMG'] = 'static/img/log/'


@app.route('/', methods=['GET'])
def api_root():
    msg = 'Image-Classification-Toolkit permite realizar una clasificación dual a partir de dos conjuntos de imágenes (positivas y negativas) con los algoritmos SVM-KNN-BPNN-CNN y Transfer Learning .'
    utils.delete_images(app.config['FOLDER_POSITIVE'])
    utils.delete_images(app.config['FOLDER_NEGATIVE'])
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
            response = jsonify({'success': False, 'msg': 'invalid-type'})
            response.status_code = 400
            pass
        return response
    response = jsonify({'success': False, 'msg': 'file-upload-is-none'})
    response.status_code = 400
    return response


@app.route('/next-form', methods=['POST'])
def next_form():
    count_images_positive = utils.count_folders(app.config['FOLDER_POSITIVE'])
    count_images_negative = utils.count_folders(app.config['FOLDER_NEGATIVE'])
    class_positive = request.form['class_positive'].upper()
    class_negative = request.form['class_negative'].upper()
    count_log_images_positive = utils.count_folders(app.config['FOLDER_IMG'] + class_positive + '/')
    count_log_images_negative = utils.count_folders(app.config['FOLDER_IMG'] + class_negative + '/')
    if count_log_images_positive < 30 and count_images_negative < 30:
        if count_images_negative > 0 and count_images_positive > 0:
            if count_images_negative > 30 and count_images_positive > 30:
                folder_positive = app.config['FOLDER_IMG'] + class_positive
                folder_negative = app.config['FOLDER_IMG'] + class_negative
                utils.create_folder(folder_positive)
                utils.create_folder(folder_negative)
                utils.move_images(app.config['FOLDER_POSITIVE'], folder_positive)
                utils.move_images(app.config['FOLDER_NEGATIVE'], folder_negative)
                return render_template('form_train.html', class_positive=class_positive, class_negative=class_negative, count_images_positive=count_images_positive, count_images_negative=count_images_negative)
            else:
                response = jsonify({'success': False, 'msg': 'minimum_thirty'})
                response.status_code = 400
        else:
            response = jsonify({'success': False, 'msg': 'minimum_zero'})
            response.status_code = 400
    else:
        return render_template('form_train.html', class_positive=class_positive, class_negative=class_negative, count_images_positive=count_log_images_positive, count_images_negative=count_log_images_negative)
    return response


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
