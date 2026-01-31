"""
Abstract base class for attribution methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - lightweight fallback
    import math

    class _NumpyLite:
        ndarray = list  # type: ignore

        @staticmethod
        def array(data, dtype=float):
            return [dtype(x) for x in data]

        @staticmethod
        def sum(values):
            return float(sum(values))

        @staticmethod
        def abs(values):
            return [abs(v) for v in values]

        @staticmethod
        def argsort(values):
            return sorted(range(len(values)), key=lambda idx: values[idx])

        @staticmethod
        def mean(values):
            vals = list(values)
            return sum(vals) / len(vals) if vals else 0.0

    np = _NumpyLite()  # type: ignore

try:
    from safetymargin.datasets.prompt_example import PromptExample, ContextSource
except ImportError:  # pragma: no cover - datapath optional
    PromptExample = Any  # type: ignore

    class ContextSource(Enum):
        """Fallback enum placeholder when datasets are unavailable."""

        pass

try:
    from safetymargin.models.base import ModelWrapper
except ImportError:  # pragma: no cover - optional dependency
    ModelWrapper = Any  # type: ignore


@dataclass
class AttributionResult:
    """
    Result from running an attribution method.
    
    Contains attribution scores indicating how much each context span
    influenced the model's output.
    
    Attributes:
        example_id: ID of the example being analyzed
        span_scores: Attribution scores for each span [num_spans]
        source_scores: Aggregated scores by source type
        token_scores: Token-level attribution scores (if available)
        predicted_output: The model's actual output
        method_name: Name of the attribution method used
        metadata: Additional method-specific information
    """
    example_id: str
    span_scores: np.ndarray
    source_scores: Dict[ContextSource, float]
    token_scores: Optional[np.ndarray] = None
    predicted_output: Optional[str] = None
    method_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_top_spans(self, k: int = 3) -> List[int]:
        """
        Get indices of top-k most influential spans.
        
        Args:
            k: Number of top spans to return
            
        Returns:
            List of span indices sorted by attribution score
        """
        return np.argsort(self.span_scores)[-k:][::-1].tolist()
    
    def get_top_sources(self, k: int = 2) -> List[ContextSource]:
        """
        Get top-k most influential context sources.
        
        Args:
            k: Number of top sources to return
            
        Returns:
            List of ContextSource enums sorted by score
        """
        sorted_sources = sorted(
            self.source_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [source for source, _ in sorted_sources[:k]]
    
    def normalize_scores(self):
        """Normalize span scores to sum to 1."""
        total = np.sum(np.abs(self.span_scores))
        if total > 0:
            self.span_scores = self.span_scores / total
    
    def __repr__(self) -> str:
        return f"AttributionResult(id={self.example_id}, method={self.method_name}, spans={len(self.span_scores)})"


class AttributionMethod(ABC):
    """
    Abstract base class for all attribution methods.
    
    Attribution methods analyze how different parts of the input context
    (system prompt, user input, history, etc.) influence the model's output.
    
    Subclasses must implement the `run` method to compute attribution scores.
    
    Attributes:
        model: The model wrapper to use for attribution
        name: Human-readable name for this method
    """
    
    def __init__(self, model: ModelWrapper, name: str = "AttributionMethod"):
        """
        Initialize the attribution method.
        
        Args:
            model: Model wrapper providing the LLM interface
            name: Name identifier for this method
        """
        self.model = model
        self.name = name
    
    @abstractmethod
    def run(self, example: PromptExample) -> AttributionResult:
        """
        Run attribution analysis on a prompt example.
        
        This method should compute attribution scores indicating how much
        each context span influences the model's output.
        
        Args:
            example: The prompt example to analyze
            
        Returns:
            AttributionResult with span and source scores
        """
        pass
    
    def aggregate_by_source(
        self,
        example: PromptExample,
        span_scores: np.ndarray,
    ) -> Dict[ContextSource, float]:
        """
        Aggregate span-level scores to source-level scores.
        
        Args:
            example: The prompt example
            span_scores: Attribution score for each span
            
        Returns:
            Dictionary mapping sources to aggregated scores
        """
        source_scores = {}
        
        for source in ContextSource:
            source_spans = [i for i, span in enumerate(example.spans) if span.source == source]
            if source_spans:
                source_scores[source] = float(np.sum(span_scores[source_spans]))
            else:
                source_scores[source] = 0.0
        
        return source_scores
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.model_name})"
