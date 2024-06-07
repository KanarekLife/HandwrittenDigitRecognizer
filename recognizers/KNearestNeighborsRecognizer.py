from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
from sklearn.neighbors import KNeighborsClassifier
import joblib
from pathlib import Path
from PIL import Image
import numpy as np

class KNearestNeighborsRecognizer():
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)
        
        if not Path('models/knn_model.pkl').exists():
            self.knn = KNeighborsClassifier()
            print("Training the KNearest Neighbors model...")
            self.knn.fit(x, y)
            Path('models').mkdir(exist_ok=True)
            joblib.dump(self.knn, 'models/knn_model.pkl')
            print("Model trained.")
        else:
            self.knn = joblib.load('models/knn_model.pkl')
        
    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.knn.predict([arr])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        return self.knn.predict(data)