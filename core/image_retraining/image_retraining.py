import os
from os import listdir
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


DELETE_TMP = 'rm -rf /tmp/output_graph.pb'
DELETE_MODELS = 'rm -rf core/image_retraining/models/'
MOVE_OUTPUT_GRAPH = 'cp /tmp/output_graph.pb core/image_retraining/models/output_graph.pb'
FOLDER_SINGLE_PREDICTION = 'core/image_retraining/single_prediction/'
FILE_OUTPUT_LABELS = 'core/image_retraining/output_labels/output_labels.txt'
FILE_OUTPUT_GRAPH = 'core/image_retraining/models/output_graph.pb'
CREATE_LOG_FILE = 'cat > core/image_retraining/models/log.txt'
FILE_LOG = 'core/image_retraining/models/log.txt'


def train(training_steps):
    os.system(DELETE_TMP)
    os.system(DELETE_MODELS)
    os.system('cat /dev/null > ' + FILE_LOG)
    os.system('python core/image_retraining/tensorflow/tensorflow/examples/image_retraining/retrain.py --saved_model_dir=core/image_retraining/models/ --image_dir=core/image_retraining/dataset/ --output_labels=core/image_retraining/output_labels/output_labels.txt --model_base_path=core/image_retraining/models/ --how_many_training_steps ' + training_steps)
    os.system(MOVE_OUTPUT_GRAPH)
    respond = {}
    respond['success'] = True
    return respond


def classification():
    images = __get_images(FOLDER_SINGLE_PREDICTION)
    ext_img = ''
    for image in images:
        ext_img = image.split('.')[1]
    image_data = tf.gfile.FastGFile(FOLDER_SINGLE_PREDICTION + 'single-prediction.' + ext_img, 'rb').read()
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(FILE_OUTPUT_LABELS)]

    with tf.gfile.FastGFile(FILE_OUTPUT_GRAPH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    respond = {}
    i = 1
    for node_id in top_k:
        human_string = label_lines[node_id]
        respond['class-' + str(i)] = human_string
        score = predictions[0][node_id]
        respond['score-' + str(i)] = '%.5f' % (score)
        i = i + 1
    respond['success'] = True
    return respond


def __get_images(folder):
    return [join(folder, file) for file in listdir(folder)]


if __name__ == '__main__':
    print(train())
    print(classification())
