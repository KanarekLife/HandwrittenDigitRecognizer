from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from PIL import Image
import numpy as np
import time
from .Recognizer import Recognizer

MODEL_PATH = Path('models/random_forest_model.pkl')

class RandomForestTreeRecognizer(Recognizer):
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)

        if not MODEL_PATH.exists():
            self.clf = RandomForestClassifier(n_jobs=-1)
            print("Training the Random Forest Tree model...")
            start_time = time.time()
            self.clf.fit(x, y)
            elapsed_time = time.time() - start_time
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(self.clf, MODEL_PATH)
            print(f"Model trained. (Elapsed time: {elapsed_time:.2f} s)")
        else:
            self.clf = joblib.load(MODEL_PATH)

    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.clf.predict([arr])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        return self.clf.predict(data)
