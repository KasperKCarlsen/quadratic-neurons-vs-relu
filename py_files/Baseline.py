import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np

from funcs import load_checkpoint, iterate_batches, resolve_amp_dtype

# torchvision.datasets.CIFAR10(root='./data', download=True) #run once


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, relu, stride=1, option="A"):
        super().__init__()
        self.relu = relu

        self.conv1r = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
        init.kaiming_normal_(self.conv1r.weight)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2r = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        init.kaiming_normal_(self.conv2r.weight)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4)
                )
            )

    def forward(self, x):
        relu = self.relu


        out = self.bn1(self.conv1r(x))
        if relu == 3:
            out = F.relu(out)
        out = self.bn2(self.conv2r(out))
        out += self.shortcut(x)
        if relu in [2, 3]:
            out = F.relu(out)
        return out


class QResNet(nn.Module):
    def __init__(self, relu, block, num_blocks, classes):
        super().__init__()
        self.relu = relu
        self.in_planes = 16
        self.classes = classes
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, len(self.classes))

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, self.relu, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.relu == 0:
            out = self.bn1(self.conv1(x))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.layer1(out))
            out = F.relu(self.layer2(out))
            out = F.relu(self.layer3(out))
    
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)

        return self.linear(out)


class Baseline:

    def __init__(
        self,
        Relu,
        num_blocks,
        block=BasicBlock,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        trial=None,
        amp_dtype="bf16",
    ):
        self.model = QResNet(Relu, block, num_blocks, classes)
        self.trial = trial
        if not torch.cuda.is_available():
            raise Exception("no cuda")
        self.device = torch.device("cuda")
        # channels_last + bf16 autocast is the standard fast recipe for conv
        # nets on Tensor Core GPUs (A100); keep plain fp32/contiguous on CPU.
        self.memory_format = torch.channels_last if self.device.type == "cuda" else torch.contiguous_format
        self.use_amp, self.amp_dtype, _needs_scaler = resolve_amp_dtype(amp_dtype, self.device)
        # fp16 (V100/Volta path) needs gradient scaling; bf16 (A100) does not.
        self.scaler = torch.amp.GradScaler("cuda", enabled=_needs_scaler)
        self.model.to(self.device, memory_format=self.memory_format)
        self.fwd = self.model
        self.criterion = nn.CrossEntropyLoss()

        if self.trial is not None:
            self.optimizer_name = self.trial.suggest_categorical(
                "optimizer", ["Adam", "AdamW", "SGD"]
            )
            self.lr = self.trial.suggest_float("lr", 1e-6, 1e-2, log=True)
            if self.optimizer_name == "AdamW":
                self.weight_decay = self.trial.suggest_float(
                    "weight_decay", 1e-5, 1e-1, log=True
                )
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, fused=self.use_amp
                )
            elif self.optimizer_name == "SGD":
                self.momentum = self.trial.suggest_float("momentum", 0.5, 0.99)
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.lr, momentum=self.momentum
                )
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, fused=self.use_amp)
        else:
            self.optimizer_name = "Adam"
            self.lr = 1e-3
            self.optimizer = None

    def train(self, X, y, batch_size, lr, epochs):
        # X/y already live on self.device (loaded resident once in
        # Train_Model.py) - iterate_batches indexes batches straight out of
        # them, skipping DataLoader/TensorDataset collation, pinning and the
        # per-batch host->device copy entirely.
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            # Accumulate on-device and only sync to the CPU once per epoch -
            # calling .item() inside the loop forces a CUDA sync every batch.
            total = torch.zeros((), device=self.device)
            correct = torch.zeros((), device=self.device)
            loss_sum = torch.zeros((), device=self.device)

            # Keep batch shape constant on GPU so cudnn.benchmark doesn't have
            # to re-tune kernels for a one-off, smaller final batch each epoch.
            for images, labels in iterate_batches(X, y, batch_size, shuffle=True, drop_last=self.use_amp):
                images = images.to(memory_format=self.memory_format)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.fwd(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_sum += loss.detach() * labels.size(0)
                correct += (outputs.detach().argmax(1) == labels).sum()
                total += labels.size(0)



        return acc_train, loss_train

    def test(self, X, y):
        self.model.eval()  # disable dropout/batchnorm training behavior

        correct = torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)

        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            # No backward pass during eval, so memory pressure is far lower
            # than training - use a bigger batch (matches the upper end of
            # the trial's batch_size search space, which already has to fit
            # during training) to cut down on kernel-launch overhead.
            for inputs, labels in iterate_batches(X, y, batch_size=512, shuffle=False, drop_last=False):
                inputs = inputs.to(memory_format=self.memory_format)

                outputs = self.fwd(inputs)
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        return (100 * correct / total).item()

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}")

    def load(self, path):
        # Cached: called once per Optuna trial when --load_path is set, so
        # avoid re-reading the same checkpoint from disk every trial.
        self.model.load_state_dict(load_checkpoint(path))
