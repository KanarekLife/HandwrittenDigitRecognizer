from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
from sklearn.preprocessing import scale
from sklearn.svm import SVC
import joblib
from pathlib import Path
from PIL import Image
import numpy as np
import time
import lzma
from .Recognizer import Recognizer

MODEL_PATH = Path('models/svm_recognizer.pkl.xz')

class NonLinearSVMRecognizer(Recognizer):
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)
        
        if not MODEL_PATH.exists():
            self.svc = SVC(kernel='rbf', C=10, gamma=0.001)
            print("Training the Non-Linear SVM model...")
            start_time = time.time()
            self.svc.fit(scale(x), y)
            elapsed_time = time.time() - start_time
            print(f"Model trained. (Elapsed time: {elapsed_time:.2f} s)")
            self.save()
        else:
            self.load()

    def save(self):
        MODEL_PATH.parent.mkdir(exist_ok=True)
        with lzma.open(MODEL_PATH, 'wb') as f:
            joblib.dump(self.svc, f)
    
    def load(self):
        with lzma.open(MODEL_PATH, 'rb') as f:
            self.svc = joblib.load(f)

    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.svc.predict([scale(arr)])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        data = scale(data)
        return self.svc.predict(data)