from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import joblib
from pathlib import Path
from PIL import Image
import numpy as np

class NonLinearSVMRecognizer():
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)
        
        x_train, x_test, y_train, y_test = train_test_split(scale(x), y, test_size=0.2, random_state=42)
        if not Path('temp/svm_recognizer.pkl').exists():
            self.svc = SVC(kernel='rbf', C=10, gamma=0.001)
            print("Training the Non-Linear SVM model...")
            self.svc.fit(x_train, y_train)
            joblib.dump(self.svc, 'temp/svm_recognizer.pkl')
            print("Model trained.")
            print(f"Accuracy: {self.svc.score(x_test, y_test)}")
        else:
            self.svc = joblib.load('temp/svm_recognizer.pkl')
        
    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.svc.predict([scale(arr)])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        return self.svc.predict(data)