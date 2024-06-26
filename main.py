import os
from torchvision import datasets
from recognizers.all_recognizers import (RandomForestTreeRecognizer, KNearestNeighborsRecognizer,
                                     NonLinearSVMRecognizer, LinearSVMRecognizer, NeuralNetworkRecognizer)
from recognizers.Recognizer import Recognizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import parse_data, parse_labels
from PIL import Image
import numpy as np
import time

training_dataset = datasets.MNIST('./data', train=True, download=True)

recognizers: dict[str, Recognizer] = {
    "Random Forest Tree": RandomForestTreeRecognizer(training_dataset),
    "KNearest Neighbors": KNearestNeighborsRecognizer(training_dataset),
    "NonLinear SVM": NonLinearSVMRecognizer(training_dataset),
    "Linear SVM": LinearSVMRecognizer(training_dataset),
    "Neural Network": NeuralNetworkRecognizer(training_dataset, 'cpu', epochs=14),
}

#Manual Tests
for file in os.listdir("test_data/"):
    if os.path.isdir("test_data/" + file):
        continue
    
    image = Image.open("test_data/" + file)
    print(f"Testing {file}")
    for name, recognizer in recognizers.items():
        print(f"{file}: {name} recognized {recognizer.recognize(image)}")

print("Manual tests (example data) done.")

#Manual tests - data from people

N = len([name for name in os.listdir("test_data/people") if os.path.isdir("test_data/people/" + name)])

print(f"Recognisers: ")
for recognizer in recognizers:
    print(f"{recognizer}", end=", ")
print("expected result")

for i in range(1, N+1):
    print(f"Person #{i}")
    for file in os.listdir(f"test_data/people/drawn_digits_{i}"):
        image = Image.open(f"test_data/people/drawn_digits_{i}/" + file)
        #print(f"Testing {file}")

        for name, recognizer in recognizers.items():
            print(f"{recognizer.recognize(image)}", end=",")
        
        tested_digit = file.split("_")
        # should be: 
        print(f"{ tested_digit[1] }")
    print()
print("Manual tests (data from people) done.")


#Automated Tests
x_train = parse_data(training_dataset)
y_train = parse_labels(training_dataset)

predictions_train = {}
for name, recognizer in recognizers.items():
    start_time = time.time()
    predictions_train[name] = recognizer.recognize_batch(x_train)
    elapsed_time = time.time() - start_time
    print(f"{name} tested on training data. (Elapsed time: {elapsed_time:.2f} s)")

print("Training predictions:")
for name, prediction in predictions_train.items():
    accuracy = sum(prediction == y_train) / len(y_train)
    print(f"{name} accuracy: {accuracy:.4f}")

test_dataset = datasets.MNIST('./data', train=False, download=True)
x_test = parse_data(test_dataset)
y_test = parse_labels(test_dataset)

predictions_test: dict[str, np.ndarray] = {}
for name, recognizer in recognizers.items():
    start_time = time.time()
    predictions_test[name] = recognizer.recognize_batch(x_test)
    elapsed_time = time.time() - start_time
    print(f"{name} tested on test data. (Elapsed time: {elapsed_time:.2f} s)")

print()

print("Test predictions:")
for name, prediction in predictions_test.items():
    accuracy = sum(prediction == y_test) / len(y_test)
    print(f"{name} accuracy: {accuracy:.4f}")

for name, prediction in predictions_test.items():
    confusion = confusion_matrix(y_test, prediction)
    plt.matshow(confusion)
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f'docs/{name.replace(" ", "")}ConfusionMatrix.png')
    plt.show()