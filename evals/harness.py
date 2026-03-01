import os

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

import selective_rope  # noqa


@register_model("fla")
class FlashLinearAttentionLMWrapper(HFLM):
    def __init__(self, **kwargs) -> FlashLinearAttentionLMWrapper:
        # TODO: provide options for doing inference with different kernels

        super().__init__(**kwargs)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Override to optionally disable KV caching for backward compatibility.

        Set FLA_DISABLE_KV_CACHE=1 to disable caching (useful for older code
        snapshots that don't have proper SelectiveRoPE caching support).
        """
        if os.environ.get("FLA_DISABLE_KV_CACHE", "0") == "1":
            # Force disable caching - remove any existing use_cache to avoid duplicate kwarg error
            generation_kwargs.pop("use_cache", None)
            generation_kwargs["use_cache"] = False
        return super()._model_generate(context, max_length, stop, **generation_kwargs)


if __name__ == "__main__":
    cli_evaluate()
