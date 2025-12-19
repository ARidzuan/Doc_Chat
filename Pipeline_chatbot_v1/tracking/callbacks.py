"""
Custom LangChain callback handler for tracking LLM usage.
Works with both LangSmith and local tracking.
"""
import time
from typing import Any, Dict, List, Optional
from langchain.callbacks.base import BaseCallbackHandler
from .local_tracker import get_tracker


class UsageTrackingCallback(BaseCallbackHandler):
    """
    Callback handler that tracks LLM usage metrics.
    Automatically logs to local SQLite database.
    Also compatible with LangSmith when enabled.
    """

    def __init__(self, model_name: str = "unknown"):
        """Initialize callback with model name."""
        super().__init__()
        self.model_name = model_name
        self.tracker = get_tracker()
        self._start_times = {}
        self._prompts = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Record when LLM call starts."""
        run_id = kwargs.get("run_id")
        if run_id:
            self._start_times[str(run_id)] = time.time()
            self._prompts[str(run_id)] = prompts[0] if prompts else None

    def on_llm_end(
        self,
        response: Any,
        **kwargs: Any
    ) -> None:
        """Record when LLM call ends successfully."""
        run_id = kwargs.get("run_id")
        if not run_id:
            return

        run_id_str = str(run_id)
        start_time = self._start_times.pop(run_id_str, None)
        prompt = self._prompts.pop(run_id_str, None)

        if start_time is None:
            return

        latency_ms = (time.time() - start_time) * 1000

        # Extract response text
        response_text = None
        if hasattr(response, 'generations') and response.generations:
            gen = response.generations[0]
            if gen and hasattr(gen[0], 'text'):
                response_text = gen[0].text

        # Estimate token counts (rough approximation)
        prompt_tokens = len(prompt.split()) * 1.3 if prompt else 0
        completion_tokens = len(response_text.split()) * 1.3 if response_text else 0

        # Extract metadata
        metadata = {}
        if hasattr(response, 'llm_output') and response.llm_output:
            metadata = response.llm_output

        self.tracker.log_call(
            model_name=self.model_name,
            operation_type="generate",
            prompt=prompt,
            response=response_text,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            latency_ms=latency_ms,
            metadata=metadata
        )

    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Record when LLM call fails."""
        run_id = kwargs.get("run_id")
        if not run_id:
            return

        run_id_str = str(run_id)
        start_time = self._start_times.pop(run_id_str, None)
        prompt = self._prompts.pop(run_id_str, None)

        latency_ms = (time.time() - start_time) * 1000 if start_time else 0

        self.tracker.log_call(
            model_name=self.model_name,
            operation_type="generate",
            prompt=prompt,
            latency_ms=latency_ms,
            error=str(error)
        )
