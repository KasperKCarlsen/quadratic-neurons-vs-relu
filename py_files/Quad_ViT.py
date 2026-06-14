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


class Quad_adapters:
    """Pretrained ViT with quadratic query/value projections (key stays linear).
    The linear terms (r) are frozen at their pretrained values; the quadratic
    terms (g, b) are trained, and the rest of the ViT is finetuned. Softmax
    attention is retained. Gradient clipping is the stability measure, applied
    identically to the ViT_baseline so the two share one training recipe."""

    def __init__(
        self,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        full=True,            # quadratic form: True = r*g + b(x^2), False = r + b(x^2)
        grad_clip=1.0,        # max grad norm; None disables
        model_name=MODEL_NAME,
        trial=None,
    ):
        self.classes = classes
        self.grad_clip = grad_clip

        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=len(classes), ignore_mismatched_sizes=True,
        )

        quadratify(self.model, full=full)

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