from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
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
        
        if not Path('models/svm_recognizer.pkl').exists():
            self.svc = SVC(kernel='rbf', C=10, gamma=0.001)
            print("Training the Non-Linear SVM model...")
            self.svc.fit(x, y)
            joblib.dump(self.svc, 'models/svm_recognizer.pkl')
            print("Model trained.")
        else:
            self.svc = joblib.load('models/svm_recognizer.pkl')
        
    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.svc.predict([scale(arr)])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        return self.svc.predict(data)