import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "./mangrove"
# DATADIR = "C:/Users/asus/Downloads/mangrove"
# C:\Users\asus\Downloads\mangrove
CATEGORIES = ["Burgeria gymnoriza", "Rhizopora mucronata", "Rhizopora stylosa"]
# CATEGORIES = ["Rhizopora stylosa", "Rizhopora mucronata"]

IMG_SIZE = 100

training_data = []

def create_training_data():
    for category in CATEGORIES:
        # print(category)
        # print(DATADIR)
        path1 = os.path.join(DATADIR, category)
        print(path1)
        class_num = CATEGORIES.index(category)
        # print(class_num)
        # print(os.listdir(path))
        # dir_list=os.listdir(path1)
        # print(os.path.getsize(path1))
        for img in os.listdir(path1):
            try:
                img_array = cv2.imread(os.path.join(path1, img), cv2.IMREAD_COLOR)
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
# print(X)
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# file_Xpickle = open("X.pickle", "rb")
# dataXpickle = pickle.load("X.pickle")
# file.close()
# print("show pickle data")
# i = 0
# for item in dataXpickle :
#     print("data", i, "adalah", item)
#     i+=1