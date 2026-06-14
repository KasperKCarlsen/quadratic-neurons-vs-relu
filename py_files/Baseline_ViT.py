import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"


class ViT_baseline:
    """Standard pretrained ViT finetuned on the target classes; the RQ3 baseline.

    freeze_qv_linear: if True, freezes the linear query/value projections so the
    ONLY difference from the quadratic model is the presence of the quadratic
    terms (the clean "do quadratic terms help" ablation). Default False = ordinary
    full finetuning.

    grad_clip: max gradient norm, applied identically to the quadratic model so
    both share one training recipe. A practical no-op for the stable baseline, but
    it keeps the comparison recipe-matched.
    """

    def __init__(
        self,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        freeze_qv_linear=False,
        grad_clip=1.0,
        model_name=MODEL_NAME,
        trial=None,
    ):
        self.classes = classes
        self.grad_clip = grad_clip

        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=len(classes), ignore_mismatched_sizes=True,
        )

        if freeze_qv_linear:
            for name, param in self.model.named_parameters():
                if ".query." in name or ".value." in name:
                    param.requires_grad = False

        self.trial = trial
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.trial is not None:
            self.optimizer_name = self.trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
            self.lr = self.trial.suggest_float("lr", 1e-6, 1e-2, log=True)
            trainable = filter(lambda p: p.requires_grad, self.model.parameters())
            if self.optimizer_name == "AdamW":
                self.weight_decay = self.trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
                self.optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer_name == "SGD":
                self.momentum = self.trial.suggest_float("momentum", 0.5, 0.99)
                self.optimizer = torch.optim.SGD(trainable, lr=self.lr, momentum=self.momentum)
            else:
                self.optimizer = torch.optim.Adam(trainable, lr=self.lr)
        else:
            self.optimizer_name = "Adam"
            self.lr = 1e-3
            self.optimizer = None

    def _resize(self, images):
        """ViT-base expects 224x224; upsample (e.g. 32x32 CIFAR) if needed."""
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return images

    def train(self, X, y, batch_size, lr, epochs):
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        self.model.to(self.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total, correct, loss_sum = 0, 0, 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = self._resize(images)

                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.model.parameters() if p.requires_grad),
                        max_norm=self.grad_clip)
                self.optimizer.step()

                loss_sum += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_sum/total:.4f} | Acc: {correct/total:.4f}")

    def test(self, X, y):
        self.model.eval()
        correct, total = 0, 0
        loader = DataLoader(TensorDataset(X, y), batch_size=128)
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = self._resize(inputs)
                outputs = self.model(pixel_values=inputs).logits
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        return 100 * correct / total

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    vb = ViT_baseline(classes=list(range(10)))
    X = torch.randn(8, 3, 32, 32)  # CIFAR-sized; _resize handles upscaling
    y = torch.randint(0, 10, (8,))
    vb.train(X, y, batch_size=4, lr=1e-3, epochs=1)
    print("test acc:", vb.test(X, y))