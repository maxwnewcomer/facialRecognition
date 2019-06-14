from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow import keras
import argparse
import pickle
import os

def train(embeddingsPath, recognizerPath, label, projectPath):

    print('[INFO] loading embeddings into model trainer...')
    data = pickle.loads(open(embeddingsPath, 'rb').read())

    embeddingsData = np.array(data['embeddings'])

    print('[INFO] encoding labels...')
    le = LabelEncoder()
    labels = le.fit_transform(data['names'])

    print('[INFO] training model...')

    model = keras.models.Sequential([
        keras.layers.Dense(512, activation = tf.nn.relu, input_shape = (128,)),
        keras.layers.Dense(128, activation = tf.nn.relu),
        keras.layers.Dense(units = sum([len(folder) for _, _, folder in os.walk(projectPath + "/dataset")]), activation = tf.nn.softmax)
    ])
    model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    checkpoint_path = projectPath + "/output/bestcheckpoint.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only = True, verbose=0)

    model.fit(embeddingsData, labels, epochs = 100, verbose = 1, validation_split= .33, use_multiprocessing = True, callbacks = [cp_callback])

    print('[INFO] writing files...')

    model.save(recognizerPath)
    f = open(label, "wb")
    f.write(pickle.dumps(le))
    f.close()
