import os
import tensorflow as tf
from flask import jsonify

DELETE_TMP = 'rm -rf /tmp/output_graph.pb'
DELETE_MODELS = 'rm -rf core/image_retraining/models/'
MOVE_OUTPUT_GRAPH = 'cp /tmp/output_graph.pb core/image_retraining/models/output_graph.pb'
FILE_SINGLE_PREDICTION = 'core/image_retraining/dataset/single-prediction.'
FILE_OUTPUT_LABELS = 'core/image_retraining/output_labels/output_labels.txt'
FILE_OUTPUT_GRAPH = 'core/image_retraining/models/output_graph.pb'
CREATE_LOG_FILE = 'cat > core/image_retraining/models/log.txt'
FILE_LOG = 'core/image_retraining/models/log.txt'


def train(training_steps):
    os.system(DELETE_TMP)
    os.system(DELETE_MODELS)
    os.system('python core/image_retraining/tensorflow/tensorflow/examples/image_retraining/retrain.py --saved_model_dir=core/image_retraining/models/ --image_dir=core/image_retraining/dataset/ --output_labels=core/image_retraining/output_labels/output_labels.txt --model_base_path=core/image_retraining/models/ --how_many_training_steps ' + training_steps)
    os.system(MOVE_OUTPUT_GRAPH)
    respond = {}
    respond['success'] = True
    return respond


def classification():
    image_data = tf.gfile.FastGFile(FILE_SINGLE_PREDICTION, 'rb').read()
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
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        respond[human_string] = '%.5f' % (score)
    respond['success'] = 'true'
    return jsonify(respond)


if __name__ == '__main__':
    print(train())
    print(classification())
