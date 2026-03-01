import os
from unittest.mock import patch

os.environ["SELECTIVE_ROPE_TESTING"] = "1"

import pytest
import torch

_ORIG_EMPTY = torch.empty
_ORIG_EMPTY_LIKE = torch.empty_like
_ORIG_NEW_EMPTY = torch.Tensor.new_empty


def _nan_empty(*args, **kwargs):
    t = _ORIG_EMPTY(*args, **kwargs)
    if t.is_floating_point():
        t.fill_(float("nan"))
    return t


def _nan_empty_like(input, **kwargs):
    t = _ORIG_EMPTY_LIKE(input, **kwargs)
    if t.is_floating_point():
        t.fill_(float("nan"))
    return t


def _nan_new_empty(self, *args, **kwargs):
    t = _ORIG_NEW_EMPTY(self, *args, **kwargs)
    if t.is_floating_point():
        t.fill_(float("nan"))
    return t


@pytest.fixture(autouse=True)
def guard_uninit_memory():
    with (
        patch("torch.empty", new=_nan_empty),
        patch("torch.empty_like", new=_nan_empty_like),
        patch("torch.Tensor.new_empty", new=_nan_new_empty),
    ):
        yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
