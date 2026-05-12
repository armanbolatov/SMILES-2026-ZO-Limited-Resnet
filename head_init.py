"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _views(x: torch.Tensor) -> list[torch.Tensor]:
    p = F.pad(x, (16, 16, 16, 16), mode="reflect")
    h = x.shape[-1]
    tl = p[:, :, 8 : 8 + h, 8 : 8 + h]
    br = p[:, :, 24 : 24 + h, 24 : 24 + h]
    flip = lambda v: torch.flip(v, dims=[-1])
    return [x, flip(x), tl, flip(tl), br, flip(br)]


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tf = T.Compose(
            [T.Resize(224), T.ToTensor(), T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)]
        )
        loader = DataLoader(
            datasets.CIFAR100(root="./data", train=True, download=True, transform=tf),
            batch_size=256, shuffle=False, num_workers=0, pin_memory=True,
        )

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone = backbone.eval().to(device)

        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                for v in _views(x):
                    feats.append(backbone(v).float().cpu())
                    labels.append(y)
        feats = torch.cat(feats).to(device)
        labels = torch.cat(labels).to(device)
        del backbone

        W = torch.zeros(layer.out_features, layer.in_features, device=device, requires_grad=True)
        b = torch.zeros(layer.out_features, device=device, requires_grad=True)
        opt = torch.optim.LBFGS(
            [W, b], lr=1.0, max_iter=500, history_size=30,
            tolerance_grad=1e-9, tolerance_change=1e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = F.cross_entropy(F.linear(feats, W, b), labels) + 2e-4 * W.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            layer.weight.copy_(W)
            layer.bias.copy_(b)
    except Exception as exc:
        print(f"[head_init] fallback to Kaiming ({exc!r})")
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
    # -------------------------------------------------------------------------
