import torch
import torch.utils.data
from torch._prims_common import DeviceLikeType
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from utils import center_image, normalize_image

MODEL_PATH = Path('models/nn_model.pth')

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

class NeuralNetworkRecognizer:
    def __init__(self, dataset: torch.utils.data.Dataset, device: DeviceLikeType, epochs: int = 10, force_retrain: bool = False):
        self.device = device
        self.network = Net().to(device)
        self.optimizer = torch.optim.Adadelta(self.network.parameters(), lr=1.0)

        if MODEL_PATH.exists() and not force_retrain:
            self.network.load_state_dict(torch.load(MODEL_PATH))
            self.network.eval()
            return

        print("Training the Neural Network model...")
        loader_prefs = {}
        if self.device == 'cuda':
            loader_prefs |= {'num_workers': 1, 'pin_memory': True}
        dataset.transform = transform
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, **loader_prefs)

        for epoch in range(epochs):
            self.train(epoch + 1, train_loader)
        
        torch.save(self.network.state_dict(), MODEL_PATH)

        print("Model trained.")

    def train(self, epoch: int, train_loader: torch.utils.data.DataLoader):
        self.network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def recognize(self, image: Image) -> int | None:
        image = center_image(normalize_image(image))
        data = transform(image).to(self.device)
        data = data.view(1, 1, 28, 28)
        output = self.network(data)
        pred = output.data.max(1, keepdim=True)[1]
        return pred.item()
