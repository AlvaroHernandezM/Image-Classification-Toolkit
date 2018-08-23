# Image Classification Toolkit

Aplicación Web que permite la clasificación dual con los algoritmos K-NN, SVM, BPNN, CNN e Image Retraining/ Transfer Learning a partir de dos conjuntos de imágenes

## [App](http://www.loencontre.co:5000)


## Requeriments

### phyton 3
```sh
sudo apt install python3
```

### python-pip
```sh
sudo apt install python-pip
```

### virtualenv
```sh
sudo pip install virtualenv
```

### python3-dev
```sh
sudo apt install python3-dev

## Run
```sh
git clone https://github.com/AlvaroHernandezM/Image-Classification-Toolkit.git
cd Image-Classification-Toolkit
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
sh create_folders.sh
git clone https://github.com/tensorflow/tensorflow.git core/image_retraining/tensorflow/


# Run app http://localhost:5000/
python app.py