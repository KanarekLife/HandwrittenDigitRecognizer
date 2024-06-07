import os
from sklearn.preprocessing import scale
from torchvision import datasets
from recognizers.KNearestNeighborsRecognizer import KNearestNeighborsRecognizer
from recognizers.RandomForestTreeRecognizer import RandomForestTreeRecognizer
from recognizers.NonLinearSVMRecognizer import NonLinearSVMRecognizer
from recognizers.LinearSVMRecognizer import LinearSVMRecognizer
from recognizers.NeuralNetworkRecognizer import NeuralNetworkRecognizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import parse_data, parse_labels, append_to_report, remove_existing_reports
from PIL import Image

training_dataset = datasets.MNIST('./data', train=True, download=True)

remove_existing_reports()
append_to_report("Training the models...\n")


recognizers = [
    RandomForestTreeRecognizer(training_dataset),
    KNearestNeighborsRecognizer(training_dataset),
    NonLinearSVMRecognizer(training_dataset),
    LinearSVMRecognizer(training_dataset),
    NeuralNetworkRecognizer(training_dataset, 'cpu', 14),
]

#Manual Tests
for file in os.listdir("test_data/"):
    image = Image.open("test_data/" + file)
    print(f"Testing {file}")
    for recognizer in recognizers:
        print(f"{file}: {recognizer.__class__.__name__} recognized {recognizer.recognize(image)}")
        append_to_report(f"{file}: {recognizer.__class__.__name__} recognized {recognizer.recognize(image)}", "testing")
        
        

#Automated Tests
test_dataset = datasets.MNIST('./data', train=False, download=True)
x_test = parse_data(test_dataset)
y_expected = parse_labels(test_dataset)

random_forest_predictions = recognizers[0].recognize_batch(x_test)
knn_predictions = recognizers[1].recognize_batch(x_test)
nonlinear_svm_predictions = recognizers[2].recognize_batch(scale(x_test))
nn_predictions = recognizers[4].recognize_batch(x_test)
linear_svm_predictions = recognizers[3].recognize_batch(scale(x_test))


plt.matshow(confusion_matrix(y_expected, random_forest_predictions))
plt.title("Random Forest Confusion Matrix")
plt.savefig('docs/RandomForestConfusionMatrix.png')
plt.show()


plt.matshow(confusion_matrix(y_expected, knn_predictions))
plt.title("KNN Confusion Matrix")
plt.savefig('docs/KNNConfusionMatrix.png')
plt.show()


plt.matshow(confusion_matrix(y_expected, nonlinear_svm_predictions))
plt.title("NonLinear SVM Confusion Matrix")
plt.savefig('docs/NonLinearSVMConfusionMatrix.png')
plt.show()

plt.matshow(confusion_matrix(y_expected, linear_svm_predictions))
plt.title("Linear SVM Confusion Matrix")
plt.savefig('docs/LinearSVMConfusionMatrix.png')
plt.show()


plt.matshow(confusion_matrix(y_expected, nn_predictions))
plt.title("Neural Network Confusion Matrix")
plt.savefig('docs/NeuralNetworkConfusionMatrix.png')
plt.show()