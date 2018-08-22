from flask import Flask, render_template, request, jsonify, url_for, redirect
from flask_cors import CORS
from core import main
from os.path import join
import utils
import json
from decimal import Decimal
import os


app = Flask(__name__)
CORS(app)
app.config['FOLDER_POSITIVE'] = 'static/img/positive/'
app.config['FOLDER_NEGATIVE'] = 'static/img/negative/'
app.config['FOLDER_IMG'] = 'static/img/log/'
app.config['FOLDER_DATASET_IMAGE_RETRAINING'] = 'core/image_retraining/dataset/'
app.config['FOLDER_NEGATIVE_TRAINING_DATASET_CNN'] = 'core/cnn/dataset/training_set/class_negative/'
app.config['FOLDER_POSITIVE_TRAINING_DATASET_CNN'] = 'core/cnn/dataset/training_set/class_positive/'
app.config['FOLDER_NEGATIVE_TEST_DATASET_CNN'] = 'core/cnn/dataset/test_set/class_negative/'
app.config['FOLDER_POSITIVE_TEST_DATASET_CNN'] = 'core/cnn/dataset/test_set/class_positive/'
app.config['FOLDER_DATASET_SVM_KNN_BPNN'] = 'core/svm_knn_bpnn/dataset/'
app.config['FOLDER_DATASET_SINGLE_PREDICTION'] = 'static/img/'
app.config['DELETE_DATASET_SINGLE_PREDICTION'] = 'rm -rf static/img/single-prediction*'
app.config['COPY_SINGLE_PREDICTION'] = 'cp static/img/'
app.config['MOVE_SINGLE_PREDICTION_CNN'] = ' core/cnn/dataset/'
app.config['MOVE_SINGLE_PREDICTION_IMAGE_RETRAINING'] = ' core/image_retraining/dataset/'
app.config['MOVE_SINGLE_PREDICTION_SVM_KNN_BPNN'] = ' core/svm_knn_bpnn/single_prediction/'
app.config['DELETE_DATASET_SINGLE_PREDICTION_CNN'] = 'rm -rf core/cnn/dataset/single-prediction*'
app.config['DELETE_DATASET_SINGLE_PREDICTION_IMAGE_RETRAINING'] ='rm -rf core/image_retraining/dataset/single-prediction*'
app.config['DELETE_DATASET_SINGLE_PREDICTION_SVN_KNN_BPNN'] = 'rm -rf core/svm_knn_bpnn/single_prediction/single-prediction*'


@app.route('/', methods=['GET'])
def api_root():
    msg = 'Image-Classification-Toolkit permite realizar una clasificación dual a partir de dos conjuntos de imágenes (positivas y negativas) con los algoritmos SVM-KNN-BPNN-CNN y Transfer Learning .'
    utils.delete_images(app.config['FOLDER_POSITIVE'])
    utils.delete_images(app.config['FOLDER_NEGATIVE'])
    return render_template('index.html', msg=msg)


@app.route('/file-upload/<type>', methods=['POST'])
def file_upload_positive(type):
    new_file = request.files.get('file', None)
    file_name = new_file.filename
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
        elif type == 'classification':
            os.system(app.config['DELETE_DATASET_SINGLE_PREDICTION'])
            new_file.save(join(app.config['FOLDER_DATASET_SINGLE_PREDICTION'], 'single-prediction.' + file_name.split('.')[1]))
            os.system(app.config['DELETE_DATASET_SINGLE_PREDICTION_CNN'])
            os.system(app.config['DELETE_DATASET_SINGLE_PREDICTION_IMAGE_RETRAINING'])
            os.system(app.config['DELETE_DATASET_SINGLE_PREDICTION_SVN_KNN_BPNN'])
            os.system(app.config['COPY_SINGLE_PREDICTION'] + 'single-prediction.' + file_name.split('.')[1] + app.config['MOVE_SINGLE_PREDICTION_CNN'] + 'single-prediction.' + file_name.split('.')[1])
            os.system(app.config['COPY_SINGLE_PREDICTION'] + 'single-prediction.' + file_name.split('.')[1] + app.config['MOVE_SINGLE_PREDICTION_IMAGE_RETRAINING'] + 'single-prediction.' + file_name.split('.')[1])
            os.system(app.config['COPY_SINGLE_PREDICTION'] + 'single-prediction.' + file_name.split('.')[1] + app.config['MOVE_SINGLE_PREDICTION_SVM_KNN_BPNN'] + 'single-prediction.' + file_name.split('.')[1])
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
    if count_log_images_positive < 30 and count_log_images_negative < 30:
        if count_images_negative > 0 and count_images_positive > 0:
            if count_images_negative > 30 and count_images_positive > 30:
                folder_positive = app.config['FOLDER_IMG'] + class_positive
                folder_negative = app.config['FOLDER_IMG'] + class_negative
                utils.create_folder(folder_positive)
                utils.create_folder(folder_negative)
                utils.move_images(app.config['FOLDER_POSITIVE'], folder_positive)
                utils.move_images(app.config['FOLDER_NEGATIVE'], folder_negative)
                utils.move_images_image_retraining(app.config['FOLDER_DATASET_IMAGE_RETRAINING'], app.config['FOLDER_POSITIVE'], app.config['FOLDER_NEGATIVE'], class_negative, class_positive)
                utils.move_images_cnn(app.config['FOLDER_POSITIVE'], app.config['FOLDER_NEGATIVE'], app.config['FOLDER_NEGATIVE_TRAINING_DATASET_CNN'], app.config['FOLDER_POSITIVE_TRAINING_DATASET_CNN'], app.config['FOLDER_NEGATIVE_TEST_DATASET_CNN'], app.config['FOLDER_POSITIVE_TEST_DATASET_CNN'])
                utils.move_images_svm_knn_bpnn(app.config['FOLDER_POSITIVE'], app.config['FOLDER_NEGATIVE'], app.config['FOLDER_DATASET_SVM_KNN_BPNN'], class_positive, class_negative)
                return render_template('form_train.html', class_positive=class_positive, class_negative=class_negative, count_images_positive=count_images_positive, count_images_negative=count_images_negative)
            else:
                return render_template('index.html', error='minimum_thirty', msg='Cada clase debe tener más de 30 imagenes', class_positive=class_positive, class_negative=class_negative, count_images_positive=count_images_positive, count_images_negative=count_images_negative)
        else:
            return render_template('index.html', error='minimum_zero', msg='Ninguna clase puede estar vacía, agrega imagenes', class_positive=class_positive, class_negative=class_negative)
    else:
        folder_positive = app.config['FOLDER_IMG'] + class_positive + '/'
        folder_negative = app.config['FOLDER_IMG'] + class_negative + '/'
        utils.move_images(folder_positive, app.config['FOLDER_POSITIVE'])
        utils.move_images(folder_negative, app.config['FOLDER_NEGATIVE'])

        utils.move_images_image_retraining(app.config['FOLDER_DATASET_IMAGE_RETRAINING'], app.config['FOLDER_POSITIVE'], app.config['FOLDER_NEGATIVE'], class_negative, class_positive)

        utils.move_images_cnn(app.config['FOLDER_POSITIVE'], app.config['FOLDER_NEGATIVE'], app.config['FOLDER_NEGATIVE_TRAINING_DATASET_CNN'], app.config['FOLDER_POSITIVE_TRAINING_DATASET_CNN'], app.config['FOLDER_NEGATIVE_TEST_DATASET_CNN'], app.config['FOLDER_POSITIVE_TEST_DATASET_CNN'])

        utils.move_images_svm_knn_bpnn(app.config['FOLDER_POSITIVE'], app.config['FOLDER_NEGATIVE'], app.config['FOLDER_DATASET_SVM_KNN_BPNN'], class_positive, class_negative)
        return render_template('form_train.html', class_positive=class_positive, class_negative=class_negative, count_images_positive=count_log_images_positive, count_images_negative=count_log_images_negative)


@app.route('/train', methods=['POST'])
def train():
    focus = 'histogram' if request.form['focus'] == 'Histograma' else 'pixel'
    n_neighbors = request.form['n_neighbors']
    max_iter_svm = request.form['max_iter_svm']
    hidden_layer_sizes = request.form['hidden_layer_sizes']
    max_iter_bpnn = request.form['max_iter_bpnn']
    epochs = request.form['epochs']
    steps_per_epoch = request.form['steps_per_epoch']
    validation_steps = request.form['validations_steps']
    training_steps = request.form['training_steps']
    class_positive = request.form['class_positive'].upper()
    class_negative = request.form['class_negative'].upper()
    response_cnn = utils.train_cnn(steps_per_epoch, epochs, validation_steps, class_positive, class_negative)
    if response_cnn['success'] == True:
        response_svm_knn_bpnn = utils.train_svm_knn_bpnn(n_neighbors, focus, hidden_layer_sizes, max_iter_bpnn, max_iter_svm)
        if response_svm_knn_bpnn['success'] == True:
            response_image_retraining = utils.train_image_retraining(training_steps)
            if response_image_retraining['success'] == True:
                return render_template('form_predict.html', val_loss_cnn=round(Decimal(response_cnn['val_loss']) * 100), val_acc_cnn=round(Decimal(response_cnn['val_acc']) * 100), loss_cnn=round(Decimal(response_cnn['loss']) * 100), acc_cnn=round(Decimal(response_cnn['acc']) * 100), positive_class=response_cnn['positive_class'], negative_class=response_cnn['negative_class'], length_images=response_svm_knn_bpnn['length_images'], size_package=response_svm_knn_bpnn['size_package'], accuracy_knn=response_svm_knn_bpnn['accuracy_knn'], accuracy_bpnn=response_svm_knn_bpnn['accuracy_bpnn'], accuracy_svm=response_svm_knn_bpnn['accuracy_svm'])
            else:
                return jsonify({'success': False, 'msg': 'error-train-image-retraining'})
        else:
            return jsonify({'success': False, 'msg': 'error-train-svm-knn-bpnn'})
    else:
        return jsonify({'success': False, 'msg': 'error-train-cnn'})


@app.route('/classification_cnn', methods=['GET'])
def classification_cnn():
    return main.classification_cnn()


@app.route('/classification_image_retraining')
def classification_image_retraining():
    return main.classification_image_retraining()


@app.route('/classification_svm_knn_bpnn')
def classification_svm_knn_bpnn():
    return main.classification_svm_knn_bpnn()


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
