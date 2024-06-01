from utils import parse_data, parse_labels, convert_from_image, append_to_report
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import joblib
from pathlib import Path
from PIL import Image
import numpy as np

class LinearSVMRecognizer():
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)
        
        x_train, x_test, y_train, y_test = train_test_split(scale(x), y, test_size=0.2, random_state=42)
        if not Path('temp/svm_recognizer_lin.pkl').exists():
            self.svc = SVC(kernel='linear')
            print("Training the Linear SVM model...")
            self.svc.fit(x_train, y_train)
            Path('temp').mkdir(exist_ok=True)
            joblib.dump(self.svc, 'temp/svm_recognizer_lin.pkl')
            print("Model trained.")
            print(f"Accuracy: {self.svc.score(x_test, y_test)}")
            append_to_report(f"Linear SVM Model Accuracy: {self.svc.score(x_test, y_test)}")
        else:
            self.svc = joblib.load('temp/svm_recognizer_lin.pkl')
        
    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.svc.predict([scale(arr)])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        return self.svc.predict(data)