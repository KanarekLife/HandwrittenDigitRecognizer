from utils import parse_data, parse_labels, convert_from_image
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from pathlib import Path
from PIL import Image
import numpy as np

class KNearestNeighborsRecognizer():
    def __init__(self, dataset: Tensor):
        x = parse_data(dataset)
        y = parse_labels(dataset)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        if not Path('temp/knn_model.pkl').exists():
            self.knn = KNeighborsClassifier()
            print("Training the model...")
            self.knn.fit(x_train, y_train)
            joblib.dump(self.knn, 'temp/knn_model.pkl')
            print("Model trained.")
            print(f"Accuracy: {self.knn.score(x_test, y_test)}")
        else:
            self.knn = joblib.load('temp/knn_model.pkl')
        
    def recognize(self, data: Image) -> int | None:
        arr = convert_from_image(data)
        return self.knn.predict([arr])[0]
    
    def recognize_batch(self, data: np.ndarray) -> int | None:
        return self.knn.predict(data)