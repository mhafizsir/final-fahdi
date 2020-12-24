import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "D:/Learning/Project/pic/Mangrove Flower"
CATEGORIES = ["Burgeria gymnoriza", "Rhizopora stylosa", "Rizhopora mucronata"]

IMG_SIZE = 100

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                if len(training_data) % 50 == 0:
                    print(f"{len(training_data)} -- progress")
            except Exception as e:
                pass


create_training_data()

print(print(f"{len(training_data)} -- completed"))

import random

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()