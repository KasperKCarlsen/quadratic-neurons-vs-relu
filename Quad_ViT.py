import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"



class QuadraticProjection(nn.Module):
    def __init__(self, in_features, out_features, bias=True, full=True):
        super().__init__()
        self.full = full
        self.scale = 1.0 

        self.r = nn.Linear(in_features, out_features, bias=bias)
        init.kaiming_normal_(self.r.weight)  

        self.b = nn.Linear(in_features, out_features, bias=bias)
        self.b.weight.data.fill_(0)
        if bias:
            self.b.bias.data.fill_(0)

        if full:
            self.g = nn.Linear(in_features, out_features, bias=bias)
            self.g.weight.data.fill_(0)
            if bias:
                self.g.bias.data.fill_(1)

    def forward(self, x):
        sq = self.scale * self.b(x * x)
        if self.full:
            return self.r(x) * self.g(x) + sq
        return self.r(x) + sq

    @classmethod
    def from_linear(cls, linear, full=True):
        has_bias = linear.bias is not None
        m = cls(linear.in_features, linear.out_features, bias=has_bias, full=full)
        with torch.no_grad():
            m.r.weight.copy_(linear.weight)
            if has_bias:
                m.r.bias.copy_(linear.bias)
        return m


def quadratify(model, full=True):
    n = 0
    for module in model.modules():
        if (
            isinstance(getattr(module, "query", None), nn.Linear)
            and isinstance(getattr(module, "value", None), nn.Linear)
        ):
            module.query = QuadraticProjection.from_linear(module.query, full)
            module.value = QuadraticProjection.from_linear(module.value, full)
            n += 1
    return n


class Quad_adapters:

    def __init__(
        self,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        full=True,            # quadratic form: True = r*g + b(x^2), False = r + b(x^2)
        model_name=MODEL_NAME,
        trial=None,
    ):
        self.classes = classes

        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(classes),
            ignore_mismatched_sizes=True, 
        )

        quadratify(self.model, full=full)


        for name, param in self.model.named_parameters():
            if ".query.r." in name or ".value.r." in name:
                param.requires_grad = False

        self.trial = trial
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

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
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            elif self.optimizer_name == "SGD":
                self.momentum = self.trial.suggest_float("momentum", 0.5, 0.99)
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.lr,
                    momentum=self.momentum,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.lr,
                )
        else:
            self.optimizer_name = "Adam"
            self.lr = 1e-3
            self.optimizer = None

    def train(self, X, y, batch_size, lr, epochs):
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
            )

        criterion = nn.CrossEntropyLoss()

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable: {trainable_params:,} / {total_params:,} params "
            f"({100*trainable_params/total_params:.1f}%)"
        )

        for epoch in range(epochs):
            self.model.train()
            total, correct, loss_sum = 0, 0, 0

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item() * labels.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {loss_sum/total:.4f} | Acc: {correct/total:.4f}"
            )

    def test(self, X, y, classes):
        self.model.eval()
        correct = 0
        total = 0

        loader = DataLoader(TensorDataset(X, y), batch_size=128)

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values=inputs).logits
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy


    def fit(self, train_loader, lr, epochs):
        self.model.to(self.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
            )
        criterion = nn.CrossEntropyLoss()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.1f}%)")

        for epoch in range(epochs):
            self.model.train()
            total, correct, loss_sum = 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_sum/total:.4f} | "
                  f"Acc: {correct/total:.4f}")

    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values=inputs).logits
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return 100 * correct / total

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))



if __name__ == "__main__":
    qa = Quad_adapters(classes=list(range(10)), full=True)

    # dummy data in the shape ViT expects
    X = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 10, (8,))

    qa.train(X, y, batch_size=4, lr=1e-3, epochs=1)
    print("test acc:", qa.test(X, y, qa.classes))