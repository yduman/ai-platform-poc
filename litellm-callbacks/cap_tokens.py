from litellm.integrations.custom_logger import CustomLogger
from litellm import token_counter
import litellm

class CapMaxTokens(CustomLogger):
    """
    Caps max_tokens to fit within model's context window.
    Prevents ContextWindowExceededError from clients that hardcode large values.
    """

    MAX_CONTEXT = 20480  # vLLM max_model_len
    MAX_OUTPUT = 8192    # Maximum output tokens to allow
    BUFFER = 256         # Safety buffer for tokenization differences

    def log_pre_api_call(self, model, messages, kwargs):
        """Modify request before it's sent to the model."""

        # Count input tokens
        try:
            input_tokens = token_counter(model=model, messages=messages)
        except Exception:
            # Fallback: estimate ~4 chars per token
            input_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4

        # Calculate available space for output
        available = self.MAX_CONTEXT - input_tokens - self.BUFFER
        available = max(100, min(available, self.MAX_OUTPUT))  # Clamp between 100 and MAX_OUTPUT

        # Get requested max_tokens
        requested = kwargs.get("max_tokens") or kwargs.get("optional_params", {}).get("max_tokens")

        if requested and requested > available:
            # Cap max_tokens
            if "optional_params" in kwargs:
                kwargs["optional_params"]["max_tokens"] = available
            kwargs["max_tokens"] = available
            print(f"[CapMaxTokens] Capped max_tokens: {requested} -> {available} (input: {input_tokens})")

        return kwargs

cap_max_tokens = CapMaxTokens()
