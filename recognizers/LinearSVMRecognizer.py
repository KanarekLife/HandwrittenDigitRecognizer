from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
import joblib
from pathlib import Path
from PIL import Image
import numpy as np
import time
from .Recognizer import Recognizer

MODEL_PATH = Path('models/svm_recognizer_lin.pkl')

class LinearSVMRecognizer(Recognizer):
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)
        
        if not MODEL_PATH.exists():
            self.svc = LinearSVC(dual='auto')
            print("Training the Linear SVM model...")
            start_time = time.time()
            self.svc.fit(scale(x), y)
            elapsed_time = time.time() - start_time
            print(f"Model trained. (Elapsed time: {elapsed_time:.2f} s)")
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(self.svc, MODEL_PATH)
        else:
            self.svc = joblib.load(MODEL_PATH)
        
    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.svc.predict([scale(arr)])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        data = scale(data)
        return self.svc.predict(data)