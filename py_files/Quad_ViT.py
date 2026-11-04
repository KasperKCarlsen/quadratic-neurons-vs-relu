import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"


class QuadraticProjection(nn.Module):
    """Hadamard quadratic neuron.
        full=True  -> r(x) * g(x) + b(x^2)
        full=False -> r(x)        + b(x^2)
    Zero-init g (weight 0, bias 1) and b (weight 0, bias 0) so at initialization
    the projection equals r(x) -- the ReLinear stability trick. from_linear copies
    a pretrained linear projection into r, so the quadratic projection starts
    exactly equal to the original ViT projection.
    """

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
    """Replace the linear query and value projections in every attention block
    with quadratic projections. The key projection is intentionally left linear:
    making it quadratic too would turn the attention logit q^T k into a product of
    two degree-2 functions (a degree-4 interaction), amplifying score magnitude."""
    n = 0
    for module in model.modules():
        if (isinstance(getattr(module, "query", None), nn.Linear)
                and isinstance(getattr(module, "value", None), nn.Linear)):
            module.query = QuadraticProjection.from_linear(module.query, full)
            module.value = QuadraticProjection.from_linear(module.value, full)
            n += 1
    return n


def _make_softmaxfree_forward(score_norm):
    """Build a replacement forward() for HF ViTSelfAttention that skips softmax.

    score_norm:
        "none"    -> raw scores: weights = QK^T / sqrt(d_head), no normalization.
        "l2"      -> each query's score row is L2-normalized over keys (a gating
                     normalizer without the exponential).
        "softmax" -> standard softmax (provided so the same module can run the
                     unmodified baseline behavior if needed).
    """
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        # Project to Q, K, V and split into heads. These mirror HF's
        # transpose_for_scores, using the module's own head config.
        def shape(x):
            new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            return x.view(*new_shape).permute(0, 2, 1, 3)

        q = shape(self.query(hidden_states))
        k = shape(self.key(hidden_states))
        v = shape(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if score_norm == "softmax":
            weights = torch.softmax(scores, dim=-1)
        elif score_norm == "l2":
            # Row-wise L2 normalization over the key axis; eps avoids /0.
            weights = scores / (scores.norm(dim=-1, keepdim=True) + 1e-6)
        else:  # "none" -> raw, unnormalized scores
            weights = scores

        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)

        return (context, weights) if output_attentions else (context, None)

    return forward


def desoftmax(model, score_norm="none"):
    """Replace the forward() of every ViTSelfAttention module so it skips softmax
    and uses `score_norm` instead. Patches in place; returns the count patched."""
    n = 0
    for module in model.modules():
        # The self-attention module is the one holding query/key/value Linears.
        if (hasattr(module, "query") and hasattr(module, "key")
                and hasattr(module, "value")
                and hasattr(module, "num_attention_heads")):
            module.forward = types.MethodType(_make_softmaxfree_forward(score_norm), module)
            n += 1
    return n


class Quad_adapters:
    """Pretrained ViT with quadratic query/value projections (key stays linear),
    and a softmax-free attention score normalization. The linear terms (r) are
    frozen at their pretrained values; the quadratic terms (g, b) are trained, and
    the rest of the ViT is finetuned. Gradient clipping is the stability measure.

    score_norm: "none" (raw scores, no normalizer), "l2" (row-L2), or "softmax".
    """

    def __init__(
        self,
        classes=[i for i in range(100)],
        full=True,
        grad_clip=1.0,
        score_norm="none",      # softmax-free by default, per supervisor's request
        model_name=MODEL_NAME,
        trial=None,
    ):
        self.classes = classes
        self.grad_clip = grad_clip
        self.score_norm = score_norm

        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=len(classes), ignore_mismatched_sizes=True,
        )

        quadratify(self.model, full=full)
        desoftmax(self.model, score_norm=score_norm)   # remove softmax from attention

        # Freeze the linear terms (r) of the quadratic Q/V projections; train the
        # quadratic terms (g, b) plus the rest of the network.
        for name, param in self.model.named_parameters():
            if ".query.r." in name or ".value.r." in name:
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
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return images

    def train(self, X, y, batch_size, lr, epochs):
        self.model.to(self.device)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Changed with AI from here
        # X, y are moved to device once, then indexed in whole batches with a
        # GPU-resident permutation. This replaces DataLoader(TensorDataset(...)),
        # which was indexing X/y one sample at a time -- each single-sample index
        # into a GPU tensor is its own CUDA kernel launch, so a 40k-sample epoch
        # was issuing tens of thousands of tiny GPU ops just to assemble batches.
        X, y = X.to(self.device), y.to(self.device)
        n = X.size(0)
        amp_enabled = self.device.type == "cuda"

        for epoch in range(epochs):
            self.model.train()
            total, correct, loss_sum = 0, 0, 0
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                images, labels = X[idx], y[idx]
                images = self._resize(images)

                self.optimizer.zero_grad()
                # bf16 autocast: ~2-3x throughput on Ampere/Ada tensor cores (e.g.
                # RTX 4070), no GradScaler needed since bf16 has fp32's exponent
                # range (unlike fp16), so it doesn't underflow small gradients.
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp_enabled):
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
            # To here
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_sum/total:.4f} | Acc: {correct/total:.4f}")

    def test(self, X, y):
        self.model.eval()
        # Changed with AI from here
        # Same fix as train(): batch via direct slicing of GPU-resident tensors
        # instead of DataLoader(TensorDataset(...)), and use bf16 autocast since
        # inference benefits from the same tensor-core speedup as training.
        X, y = X.to(self.device), y.to(self.device)
        n = X.size(0)
        batch_size = 128
        amp_enabled = self.device.type == "cuda"
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, n, batch_size):
                inputs = X[start:start + batch_size]
                labels = y[start:start + batch_size]
                inputs = self._resize(inputs)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    outputs = self.model(pixel_values=inputs).logits
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        # To here
        return 100 * correct / total

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    qa = Quad_adapters(classes=list(range(100)), full=True, score_norm="none")
    X = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    qa.train(X, y, batch_size=4, lr=1e-3, epochs=1)
    print("test acc:", qa.test(X, y))