"""
HuggingFace Transformers model wrapper implementation.
"""

import os
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

from safetymargin.models.base import ModelWrapper, ModelOutput


@dataclass
class TeacherForcingBatch:
    """
    Container for teacher-forced inputs used in attribution computations.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_rendered: str
    prompt_token_count: int
    target_token_count: int
    content_start: int


class HuggingFaceModelWrapper(ModelWrapper):
    """
    Wrapper for HuggingFace Transformers models.
    
    Provides unified interface for causal language models with support for:
    - Efficient generation with KV caching
    - Token-level log probability computation
    - Gradient-based attribution via backpropagation
    - 8-bit and 4-bit quantization for memory efficiency
    
    Example:
        >>> model = HuggingFaceModelWrapper("meta-llama/Meta-Llama-3-8B")
        >>> model.load()
        >>> output = model.generate("Hello, how are you?")
        >>> print(output.generated_text)
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        use_cache: bool = True,
        quantization_config: Any = None,
        torch_dtype: str = "auto",
        use_quantization: bool = None,
    ):
        """
        Initialize HuggingFace model wrapper.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            max_length: Maximum sequence length
            use_cache: Whether to use KV caching
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
            torch_dtype: PyTorch dtype (auto, float16, float32, bfloat16)
            use_quantization: Whether to use quantization (None=auto-detect based on CUDA availability)
        """
        super().__init__(model_name, device, max_length, use_cache)
        
        # Auto-detect quantization based on CUDA availability if not specified
        if use_quantization is None:
            use_quantization = torch.cuda.is_available()
        self.use_quantization = use_quantization
        self.quantization_config = quantization_config
        self.torch_dtype = torch_dtype
    
    def load(self):
        """Load the HuggingFace model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        # Get HuggingFace token from environment if available
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        # Determine dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.torch_dtype, "auto")
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        # Prefer left-side padding/truncation for decoder-only models so we keep the
        # most recent tokens (including the assistant header) when long prompts occur.
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass
        try:
            # Keep the end of the prompt (assistant preamble) intact
            setattr(self.tokenizer, "truncation_side", "left")
        except Exception:
            pass
        # Use model_max_length if it looks sane; some tokenizers set an extremely large sentinel.
        try:
            t_max = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(t_max, int) and 0 < t_max < 100_000:
                # Keep the smaller of configured and tokenizer limits
                self.max_length = min(self.max_length, t_max) if self.max_length else t_max
        except Exception:
            pass
        # Set padding token if not set (and suppress warning)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "token": hf_token,
            "attn_implementation": "eager",
        }
        # Use quantization_config if provided
        if self.use_quantization and self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            # CPU mode or no quantization, use float32 for compatibility
            model_kwargs["torch_dtype"] = torch.float32 if torch_dtype == "auto" else torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        # Move to device if not using device_map/quantization
        if not (self.use_quantization and self.quantization_config is not None):
            self.model = self.model.to(self.device)
        # Ensure attentions and hidden states are available for downstream attribution
        self.model.config.output_attentions = True
        self.model.config.output_hidden_states = True
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    @staticmethod
    def _looks_like_chat_rendered(text: str) -> bool:
        """Heuristically detect if `text` already contains a chat template rendering.

        This avoids double-applying `apply_chat_template` when callers pass a
        pre-rendered prompt. We look for special header markers used by modern
        chat templates (e.g., Llama 3).
        """
        if not isinstance(text, str) or not text:
            return False
        markers = (
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|eom_id|>",
        )
        return any(m in text for m in markers)
    
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
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer not initialised; call load() before generate().")

        use_chat_template = hasattr(tokenizer, "apply_chat_template") and not self._looks_like_chat_rendered(prompt)
        rendered_prompt = prompt
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered_prompt = prompt
                use_chat_template = False

        # Tokenize input (avoid double special tokens when chat template already adds them)
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.max_length,
        }
        if use_chat_template:
            tokenizer_kwargs["add_special_tokens"] = False

        inputs = tokenizer(
            rendered_prompt,
            **tokenizer_kwargs,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id

        # Build a safe GenerationConfig using only supported keys to avoid warnings
        gen_cfg = None
        try:
            gen_cfg = self.model.generation_config.clone()
            # Respect max_new_tokens; fall back to max_length if needed
            if hasattr(gen_cfg, "max_new_tokens"):
                gen_cfg.max_new_tokens = int(max_new_tokens)
            else:
                # some very old versions only use max_length
                gen_cfg.max_length = int(input_ids.shape[1] + max_new_tokens)
            # Enable sampling only when temperature > 0 and set related params then
            do_sample = bool((temperature or 0.0) > 0.0)
            if hasattr(gen_cfg, "do_sample"):
                gen_cfg.do_sample = do_sample
            if do_sample and hasattr(gen_cfg, "temperature"):
                gen_cfg.temperature = float(max(1e-8, temperature))
            if do_sample and hasattr(gen_cfg, "top_p"):
                gen_cfg.top_p = float(top_p)
        except Exception:
            gen_cfg = None

        # Generate
        with torch.no_grad():
            if gen_cfg is not None:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=gen_cfg,
                    use_cache=self.use_cache,
                    return_dict_in_generate=True,
                    output_scores=return_log_probs,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
            else:  # fallback to kwargs path
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    use_cache=self.use_cache,
                    return_dict_in_generate=True,
                    output_scores=return_log_probs,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

        # Extract sequences robustly across transformers versions
        sequences = getattr(outputs, "sequences", None)
        if sequences is None:
            sequences = getattr(outputs, "generated_ids", None)
        if sequences is None:
            try:
                import torch as _torch
                if isinstance(outputs, _torch.Tensor):
                    sequences = outputs
            except Exception:
                sequences = None
        if sequences is None:
            # Give up gracefully
            gen_ids = None
            gen_text = ""
        else:
            gen_ids = sequences[:, input_ids.shape[1]:]
            try:
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            except Exception:
                gen_text = ""
        
        # Compute token log probabilities if requested
        token_log_probs = None
        if return_log_probs and hasattr(outputs, 'scores'):
            # outputs.scores is a tuple of tensors, one per generation step
            scores = torch.stack(outputs.scores, dim=1)  # [batch, gen_len, vocab]
            log_probs = F.log_softmax(scores, dim=-1)
            
            # Get log prob of selected token at each step
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=generated_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch, gen_len]
        
        # Best-effort extract last-step logits if available
        last_scores = None
        if hasattr(outputs, "scores"):
            try:
                scores_obj = outputs.scores
                if isinstance(scores_obj, (list, tuple)) and len(scores_obj) > 0:
                    last_scores = scores_obj[-1]
            except Exception:
                last_scores = None

        num_generated_tokens = int(gen_ids.shape[1]) if gen_ids is not None else 0
        return ModelOutput(
            logits=last_scores,
            generated_ids=gen_ids,
            generated_text=gen_text,
            token_log_probs=token_log_probs,
            metadata={
                "num_generated_tokens": num_generated_tokens,
                "prompt_length": input_ids.shape[1],
                "prompt_preview": (rendered_prompt[:80] if isinstance(rendered_prompt, str) else ""),
            }
        )
    
    def build_teacher_forcing_batch(
        self,
        prompt_text: str,
        target_text: str,
    ) -> TeacherForcingBatch:
        """
        Construct batched tensors for teacher-forced scoring.
        """
        if target_text is None or target_text == "":
            raise ValueError("Target text must be provided for teacher forcing.")
        
        tokenizer = self.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        messages = [{"role": "user", "content": prompt_text}]
        prompt_rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        prompt_enc = tokenizer(
            prompt_rendered,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        target_enc = tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        input_ids = torch.cat(
            [prompt_enc["input_ids"], target_enc["input_ids"]],
            dim=1,
        ).to(self.device)
        attention_mask = torch.cat(
            [prompt_enc["attention_mask"], target_enc["attention_mask"]],
            dim=1,
        ).to(self.device)
        
        labels = input_ids.clone()
        prompt_token_count = prompt_enc["input_ids"].shape[1]
        labels[:, :prompt_token_count] = -100
        
        target_token_count = labels.shape[1] - prompt_token_count
        
        content_start = prompt_rendered.find(prompt_text)
        if content_start < 0:
            raise ValueError("Prompt content not found inside chat template rendering.")
        
        return TeacherForcingBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            prompt_rendered=prompt_rendered,
            prompt_token_count=prompt_token_count,
            target_token_count=target_token_count,
            content_start=content_start,
        )
    
    @staticmethod
    def _gather_target_log_probs(
        logits: torch.Tensor,
        batch: TeacherForcingBatch,
    ) -> torch.Tensor:
        """
        Gather log probabilities corresponding to target tokens.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        shifted_log_probs = log_probs[:, :-1, :]
        shifted_labels = batch.input_ids[:, 1:]
        target_mask = batch.labels[:, 1:] != -100
        
        gathered = shifted_log_probs.gather(
            dim=-1,
            index=shifted_labels.unsqueeze(-1),
        ).squeeze(-1)
        
        target_log_probs = gathered.masked_select(target_mask)
        return target_log_probs
    
    def teacher_forcing_forward(
        self,
        prompt_text: str,
        target_text: str,
        require_grad: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a teacher-forced forward pass and collect useful statistics.
        """
        batch = self.build_teacher_forcing_batch(prompt_text, target_text)
        
        model_inputs = {
            "attention_mask": batch.attention_mask,
            "labels": batch.labels,
        }
        
        if require_grad:
            outputs = self.model(
                input_ids=batch.input_ids,
                **model_inputs,
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch.input_ids,
                    **model_inputs,
                )
        
        target_log_probs = self._gather_target_log_probs(outputs.logits, batch)
        sum_log_prob = float(target_log_probs.sum().item())
        mean_log_prob = sum_log_prob / batch.target_token_count if batch.target_token_count else 0.0
        
        return {
            "batch": batch,
            "outputs": outputs,
            "target_log_probs": target_log_probs,
            "sum_log_prob": sum_log_prob,
            "mean_log_prob": mean_log_prob,
        }
    
    def get_log_probs(
        self,
        prompt: str,
        continuation: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute log probabilities for tokens.
        
        Args:
            prompt: Input prompt text
            continuation: Optional continuation text to score
            
        Returns:
            Tensor of log probabilities
        """
        if continuation is not None:
            result = self.teacher_forcing_forward(prompt, continuation)
            return result["target_log_probs"]
        
        # Fallback: compute log probs over prompt tokens only
        full_text = prompt
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs[:, :-1, :],
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs[0]
    
    # Gradient-based APIs removed per request to drop gradient code.
    
    def encode(self, text: str) -> torch.Tensor:
        """Tokenize text and return token IDs."""
        return self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)[0]
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class HFModel(HuggingFaceModelWrapper):
    """
    Thin wrapper exposing a `score` API expected by newer attribution utilities.

    The score is defined as the mean log probability of the model's generated
    continuation when prompted with the provided context plus question.
    """

    def __init__(self, model_name: str, auto_load: bool = True, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        if auto_load:
            self.load()

    def score(self, Q: str, context: List[str]) -> float:
        if self.model is None:
            self.load()
        prompt_sections = list(context) + [Q]
        prompt = "\n\n".join(section.strip() for section in prompt_sections if section)
        output = self.generate(
            prompt,
            max_new_tokens=32,
            temperature=0.0,
            return_log_probs=True,
        )
        if output.token_log_probs is None:
            return 0.0
        token_lp = output.token_log_probs
        if token_lp.ndim > 1:
            token_lp = token_lp.mean(dim=0)
        return float(token_lp.mean().item())
