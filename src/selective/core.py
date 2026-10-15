import torch
from torch import nn
from torch.nn import functional as F

from .vision import classification_metrics


METHODS = [
    "selective",
    "erm",
    "always_consistent",
    "random_gate",
    "no_uncertainty",
    "no_risk_control",
]
EVALUATION_NEEDS_GRAD = True


class Model(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, cfg["classes"])
        self.teacher = None
        if method in ("selective", "random_gate", "no_uncertainty", "no_risk_control"):
            payload = torch.load(cfg["gate_checkpoint"], map_location="cpu", weights_only=False)
            self.teacher = Model(cfg, "erm")
            self.teacher.load_state_dict(payload["teacher"])
            self.gates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(192 + cfg["transform_dimensions"], 64),
                        nn.GELU(),
                        nn.Linear(64, 1),
                    )
                    for _ in payload["gates"]
                ]
            )
            for gate, state in zip(self.gates, payload["gates"]):
                gate.load_state_dict(state)
            for parameter in list(self.teacher.parameters()) + list(self.gates.parameters()):
                parameter.requires_grad_(False)
            self.threshold = payload["threshold"]
            self.gate_coverage = payload["coverage"]

    def forward(self, x):
        embedding = self.encoder(x.float()).flatten(1)
        return self.head(embedding), embedding

    def acceptance(self, batch, cfg):
        if self.method == "erm":
            return torch.zeros(len(batch["x"]), device=batch["x"].device)
        if self.method == "always_consistent":
            return torch.ones(len(batch["x"]), device=batch["x"].device)
        if self.method == "random_gate":
            return (
                torch.rand(len(batch["x"]), device=batch["x"].device) < self.gate_coverage
            ).float()
        with torch.no_grad():
            self.teacher.eval()
            _, clean = self.teacher(batch["x"])
            _, transformed = self.teacher(batch["x_pair"])
            features = torch.cat(
                [clean, transformed, clean - transformed, batch["transform"].float()], -1
            )
            scores = torch.stack([gate(features).sigmoid().squeeze(-1) for gate in self.gates])
            mean, uncertainty = scores.mean(0), scores.std(0, unbiased=False)
            value = (
                mean
                if self.method == "no_uncertainty"
                else mean - cfg["uncertainty_weight"] * uncertainty
            )
            threshold = 0.5 if self.method == "no_risk_control" else self.threshold
            return (value >= threshold).float()


def pair_loss(left, right):
    p, q = left.softmax(-1), right.softmax(-1)
    mean = (p + q) / 2
    return 0.5 * (
        (p * (p.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
        + (q * (q.clamp_min(1e-8).log() - mean.clamp_min(1e-8).log())).sum(-1)
    )


def loss(model, batch, cfg):
    logits, _ = model(batch["x"])
    transformed, _ = model(batch["x_pair"])
    gate = model.acceptance(batch, cfg)
    return (
        F.cross_entropy(logits, batch["y"].long())
        + cfg["consistency_weight"] * (gate * pair_loss(logits, transformed)).mean()
    )


def predict(model, batch, cfg, calibration=None):
    logits, _ = model(batch["x"])
    transformed, _ = model(batch["x_pair"])
    gate = model.acceptance(batch, cfg)
    losses = pair_loss(logits, transformed)
    norms = []
    parameters = [p for p in model.parameters() if p.requires_grad]
    for value in losses:
        gradient = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        norm = sum(g.square().sum() for g in gradient if g is not None).sqrt()
        norms.append(norm)
    return {
        "logits": logits,
        "accepted": gate[:, None],
        "consistency_gradient_norm": torch.stack(norms)[:, None],
    }


def metrics(prediction, data, cfg):
    result = classification_metrics(prediction["logits"], data["y"])
    accepted = prediction["accepted"].ravel() > 0.5
    valid = data["valid"].astype(bool)
    gradients = prediction["consistency_gradient_norm"].ravel()
    result["coverage"] = float(accepted.mean())
    result["false_acceptance_risk"] = float((~valid[accepted]).mean()) if accepted.any() else None
    result["false_rejection"] = float((~accepted[valid]).mean()) if valid.any() else None
    denominator = (accepted * gradients).sum()
    result["iigr"] = (
        float((accepted * (~valid) * gradients).sum() / denominator) if denominator else None
    )
    return result


def inference(model, batch, cfg):
    return model(batch["x"])
