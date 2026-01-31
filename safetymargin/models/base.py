"""
Abstract base class for model wrappers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Protocol, Union
import torch


@dataclass
class ModelOutput:
    """
    Standardized output from a model forward pass.
    
    Attributes:
        logits: Raw logits from the model [batch_size, seq_len, vocab_size]
        log_probs: Log probabilities for each token [batch_size, seq_len, vocab_size]
        generated_ids: Token IDs for generated text [batch_size, gen_len]
        generated_text: Decoded generated text
        hidden_states: Hidden states from each layer (if available)
        attentions: Attention weights (if available)
        past_key_values: Cached key-value pairs for generation
        token_log_probs: Log probability of each generated token [batch_size, gen_len]
        metadata: Additional model-specific information
    """
    logits: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    generated_text: Optional[str] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Any] = None
    token_log_probs: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    This interface provides a unified API for interacting with different
    LLM backends (HuggingFace, OpenAI, etc.) for attribution analysis.
    
    Key features:
    - Token-level log probability computation
    - KV caching for efficient generation
    - Span-level masking/intervention
    
    Attributes:
        model_name: Identifier for the model
        device: Compute device (cuda/cpu)
        max_length: Maximum sequence length
        use_cache: Whether to use KV caching
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        use_cache: bool = True,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Model identifier (e.g., "meta-llama/Meta-Llama-3-8B")
            device: Device to load the model on
            max_length: Maximum sequence length
            use_cache: Whether to use KV caching
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.use_cache = use_cache
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_log_probs: bool = True,
    ) -> ModelOutput:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            return_log_probs: Whether to return token log probabilities
            
        Returns:
            ModelOutput with generated text and metadata
        """
        pass
    
    @abstractmethod
    def get_log_probs(
        self,
        prompt: str,
        continuation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for tokens.
        
        If continuation is provided, compute log probs for continuation tokens
        given the prompt. Otherwise, compute log probs for all prompt tokens.
        
        Args:
            prompt: Input prompt text
            continuation: Optional continuation text to score
            
        Returns:
            Tensor of log probabilities [seq_len] or [cont_len]
        """
        pass
    
    
    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenize text and return token IDs.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tensor of token IDs [seq_len]
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Decoded text string
        """
        pass
    
    def get_span_indices(
        self,
        full_text: str,
        span_text: str,
        span_start_char: int,
    ) -> Tuple[int, int]:
        """
        Get token indices for a character span.
        
        This is useful for identifying which tokens correspond to a
        specific context span for attribution analysis.
        
        Args:
            full_text: The complete text
            span_text: The span text to locate
            span_start_char: Character index where span starts
            
        Returns:
            Tuple of (start_token_idx, end_token_idx)
        """
        # Tokenize the full text
        full_ids = self.encode(full_text)
        
        # Tokenize text before and including span
        before_span = full_text[:span_start_char]
        before_ids = self.encode(before_span)
        start_idx = len(before_ids)
        
        # Tokenize including span
        with_span = full_text[:span_start_char + len(span_text)]
        with_span_ids = self.encode(with_span)
        end_idx = len(with_span_ids)
        
        return start_idx, end_idx
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"


class BaseModel(Protocol):
    """
    Minimal protocol for attribution models exposing a scalar scoring API.
    """

    def score(self, Q: str, context: List[str]) -> Union[float, Dict[str, Any]]:
        ...
