from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from PIL import Image
import numpy as np

class RandomForestTreeRecognizer():
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)
        
        if not Path('models/random_forest_model.pkl').exists():
            self.clf = RandomForestClassifier()
            print("Training the Random Forest Tree model...")
            self.clf.fit(x, y)
            Path('models').mkdir(exist_ok=True)
            joblib.dump(self.clf, 'models/random_forest_model.pkl')
            print("Model trained.")
        else:
            self.clf = joblib.load('models/random_forest_model.pkl')
        
    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.clf.predict([arr])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        return self.clf.predict(data)
