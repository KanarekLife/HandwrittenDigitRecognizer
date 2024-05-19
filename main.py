import os
from sklearn.preprocessing import scale
from torchvision import datasets
from KNearestNeighborsRecognizer import KNearestNeighborsRecognizer
from RandomForestTreeRecognizer import RandomForestTreeRecognizer
from NonLinearSVMRecognizer import NonLinearSVMRecognizer
from LinearSVMRecognizer import LinearSVMRecognizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import parse_data, parse_labels
from PIL import Image

training_dataset = datasets.MNIST('./data', train=True, download=True)

recognizers = [
    RandomForestTreeRecognizer(training_dataset),
    KNearestNeighborsRecognizer(training_dataset),
    NonLinearSVMRecognizer(training_dataset),
    LinearSVMRecognizer(training_dataset)
]

#Manual Tests

for file in os.listdir("test_data/"):
    image = Image.open("test_data/" + file)
    print(f"Testing {file}")
    for recognizer in recognizers:
        print(f"{file}: {recognizer.__class__.__name__} recognized {recognizer.recognize(image)}")

# #Automated Tests
# test_dataset = datasets.MNIST('./data', train=False, download=True)
# x_test = parse_data(test_dataset)
# y_expected = parse_labels(test_dataset)

# random_forest_predictions = recognizers[0].recognize_batch(x_test)
# knn_predictions = recognizers[1].recognize_batch(x_test)
# svm_predictions = recognizers[2].recognize_batch(scale(x_test))

# plt.matshow(confusion_matrix(y_expected, random_forest_predictions))
# plt.title("Random Forest Confusion Matrix")
# plt.show()
# plt.matshow(confusion_matrix(y_expected, knn_predictions))
# plt.title("KNN Confusion Matrix")
# plt.show()
# plt.matshow(confusion_matrix(y_expected, svm_predictions))
# plt.title("SVM Confusion Matrix")
# plt.show()
