import random

import numpy as np
import torch

from src.seed import set_seed


def draw() -> tuple[float, float, float]:
    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1).item()),
    )


def test_set_seed_makes_every_generator_repeatable() -> None:
    set_seed(7)
    first = draw()
    set_seed(7)
    second = draw()

    assert first == second


def test_different_seeds_give_different_draws() -> None:
    set_seed(1)
    first = draw()
    set_seed(2)
    second = draw()

    assert first != second
