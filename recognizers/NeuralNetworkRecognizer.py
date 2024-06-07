import torch
import torch.utils.data
from torch._prims_common import DeviceLikeType
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
from utils import center_image, normalize_image
from .Recognizer import Recognizer

MODEL_PATH = Path('models/nn_model.pth')
MAX_BATCH_SIZE = 10000

transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class NeuralNetworkRecognizer(Recognizer):
    def __init__(self, dataset: torch.utils.data.Dataset, device: DeviceLikeType, epochs: int = 10, force_retrain: bool = False):
        self.device = device
        self.network = Net().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adadelta(self.network.parameters(), lr=1.0)

        if MODEL_PATH.exists() and not force_retrain:
            self.network.load_state_dict(torch.load(MODEL_PATH))
            self.network.eval()
            return

        print("Training the Neural Network model...")
        loader_prefs = {}
        if self.device == 'cuda':
            loader_prefs |= {'num_workers': 5, 'pin_memory': True}
        dataset.transform = transform
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, **loader_prefs)

        for epoch in range(epochs):
            self.train(epoch + 1, train_loader)
        
        print("Model trained.")

        torch.save(self.network.state_dict(), MODEL_PATH)

    def train(self, epoch: int, train_loader: torch.utils.data.DataLoader):
        self.network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    

    def recognize(self, image: Image) -> int | None:
        with torch.no_grad():
            image = center_image(normalize_image(image))
            data = transform(image).to(self.device)
            data = data.view(1, 1, 28, 28)
            output = self.network(data)
            pred = output.data.max(1, keepdim=True)[1]
            return pred.item()

    def recognize_batch(self, data: np.ndarray) -> np.ndarray:
        result = np.empty(data.shape[0], dtype=int)
        for i in range(0, data.shape[0], MAX_BATCH_SIZE):
            result[i:i+MAX_BATCH_SIZE] = self.__recognize_batch_internal(data[i:i+MAX_BATCH_SIZE])
        return result

    def __recognize_batch_internal(self, data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            data = data.astype(np.float32)
            data = torch.from_numpy(data).to(self.device)
            data = data.view(-1, 1, 28, 28)
            output = self.network(data)
            pred = output.data.max(1, keepdim=True)[1]
            return pred.cpu().numpy().flatten()