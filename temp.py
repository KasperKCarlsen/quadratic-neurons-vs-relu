import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader


class QuadNeuronNet(nn.Module):
    class LambdaLayer(nn.Module):
        def __init__(self, lambd):
            super().__init__()
            self.lambd = lambd

        def forward(self, x):
            return self.lambd(x)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, relu_mode, stride=1):
            super().__init__()
            self.relu_mode = relu_mode

            self.conv1r = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
            init.kaiming_normal_(self.conv1r.weight)

            self.conv1g = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
            self.conv1g.weight.data.fill_(0)
            self.conv1g.bias.data.fill_(1)

            self.conv1b = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
            self.conv1b.weight.data.fill_(0)
            self.conv1b.bias.data.fill_(0)

            self.bn1 = nn.BatchNorm2d(planes)

            self.conv2r = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
            init.kaiming_normal_(self.conv2r.weight)

            self.conv2g = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
            self.conv2g.weight.data.fill_(0)
            self.conv2g.bias.data.fill_(1)

            self.conv2b = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
            self.conv2b.weight.data.fill_(0)
            self.conv2b.bias.data.fill_(0)

            self.bn2 = nn.BatchNorm2d(planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = QuadNeuronNet.LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4)
                    )
                )

        def forward(self, x):
            out = self.bn1(
                self.conv1r(x) * self.conv1g(x) + self.conv1b(x.pow(2))
            )

            if self.relu_mode == 2:
                out = F.relu(out)

            out = self.bn2(
                self.conv2r(out) * self.conv2g(out) + self.conv2b(out.pow(2))
            )
            out += self.shortcut(x)

            if self.relu_mode in [1, 2]:
                out = F.relu(out)

            return out

    def __init__(self, blocks=(3, 3, 3), relu_mode=1, num_classes=10):
        super().__init__()
        self.blocks = list(blocks)
        self.relu_mode = relu_mode
        self.num_classes = num_classes
        self.in_planes = 16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, self.blocks[0], stride=1)
        self.layer2 = self._make_layer(32, self.blocks[1], stride=2)
        self.layer3 = self._make_layer(64, self.blocks[2], stride=2)

        self.linear = nn.Linear(64, self.num_classes)

        self.to(self.device)

    def _make_layer(self, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(
                QuadNeuronNet.BasicBlock(
                    self.in_planes,
                    planes,
                    relu_mode=self.relu_mode,
                    stride=s
                )
            )
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.relu_mode == 0:
            out = self.bn1(self.conv1(x))
        else:
            out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)

        return self.linear(out)

    @staticmethod
    def load_batch(path):
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

        images = batch[b"data"]
        labels = batch[b"labels"]

        images = images.reshape(-1, 3, 32, 32)
        return torch.from_numpy(images).float() / 255.0, torch.tensor(labels, dtype=torch.long)

    def train_model(self, learning_rate, epochs, data, batch_size):
        """
        data: tuple (X, y)
              X should be a torch tensor of shape [N, C, H, W]
              y should be a torch tensor of labels
        """
        X, y = data
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.train()
            total = 0
            correct = 0
            loss_sum = 0.0

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {loss_sum / total:.4f} | "
                f"Acc: {correct / total:.4f}"
            )

    def test_model(self, test_data):
        """
        test_data: tuple (X, y)
        """
        X, y = test_data
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=128,
            shuffle=False
        )

        self.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100.0 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        return accuracy

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, f"{name}.pth")
        torch.save(self.state_dict(), full_path)
        print(f"Model saved to: {full_path}")

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.to(self.device)
        self.eval()
        print(f"Model loaded from: {path}")
        return self