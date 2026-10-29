"""Export ImageNet weights into a bundle: ``python -m src.deep_learning.export_backbone``.

This is the only code that touches the network. Run it once on a development
machine and ship the resulting folder with the model artifacts, never download
weights at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import models

from src.deep_learning.backbones import BACKBONES, DEFAULT_BACKBONE
from src.deep_learning.bundle import write_bundle

_IMAGENET_WEIGHTS = {
    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone", choices=sorted(BACKBONES), default=DEFAULT_BACKBONE
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="bundle folder to write"
    )
    args = parser.parse_args(argv)

    weights = _IMAGENET_WEIGHTS[args.backbone]
    model = getattr(models, args.backbone)(weights=weights)
    manifest = write_bundle(args.out, args.backbone, model.state_dict(), weights.name)
    print(f"wrote {args.out} ({manifest.backbone}, sha256 {manifest.sha256})")


if __name__ == "__main__":
    main()
