"""
SceneLLM LoRA implementation for SceneLLM.
Credit to the authors of the original code: https://doi.org/10.1016/j.patcog.2025.111992.
"""

import torch
import torch.nn as nn

# TODO: Add better LLM
try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, BitsAndBytesConfig

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/peft not available. LLM will use placeholder.")


class SceneLLMLoRA(nn.Module):
    def __init__(
        self,
        model_name,  # meta-llama/Llama-2-7b-hf google/gemma-2-2b google/gemma-3-270m
        fallback_dim=None,  # Required when transformers is not available
        r=16,
        alpha=32,
        dropout=0.05,  # why so low?
    ):  # gemma-3-270m
        """
        Implementation of SceneLLM paper with fallback support.
        # TODO: gemma, oss, llama2-4,
        Args:
            model_name (str, optional): _description_. Defaults to "meta-llama/Llama-2-7b-hf".
            fallback_dim (int, optional): Dimension for fallback linear layer when transformers unavailable.
            r (int, optional): _description_. Defaults to 16.
            alpha (int, optional): _description_. Defaults to 32.
            dropout (_type_, optional): _description_. Defaults to 0.05#whysolow?.
        """
        super().__init__()
        if TRANSFORMERS_AVAILABLE and model_name is not None:
            # Option 1: Use modern quantization config (recommended for memory efficiency)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=False,
            )

            # Option 2: Disable quantization completely (uncomment if you have enough VRAM)
            # quantization_config = None

            base = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                # device_map="auto",
                # attn_implementation="flash_attention_2"
            )
            cfg = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(base, cfg)
            self.hidden_size = base.config.hidden_size
            self.use_fallback = False
        else:
            # Use fallback linear layer
            if fallback_dim is None:
                raise ValueError(
                    "fallback_dim is required when transformers is not available or model_name is None"
                )
            self.model = nn.Linear(fallback_dim, fallback_dim)
            self.hidden_size = fallback_dim
            self.use_fallback = True

    def forward(self, token_embeds):  # [B, T, D] embeddings
        if self.use_fallback:
            # Simple linear transformation for fallback
            output = self.model(token_embeds)
            # Ensure no NaN in output
            if torch.isnan(output).any():
                print("WARNING: NaN detected in fallback LLM, using zero output")
                return torch.zeros_like(output)
            return output
        else:
            # For embeddings input, use inputs_embeds parameter
            # Make sure we only pass the expected parameters
            try:
                outputs = self.model(inputs_embeds=token_embeds)
                result = outputs.last_hidden_state  # [B, T, D]

                # Check for NaN in output and clamp extreme values
                if torch.isnan(result).any():
                    print("WARNING: NaN detected in LLM output, using zero output")
                    return torch.zeros_like(result)

                # Clamp extreme values to prevent overflow
                result = torch.clamp(result, min=-10.0, max=10.0)
                return result

            except Exception as e:
                print(f"WARNING: Error in LLM forward pass: {e}, using zero output")
                # Return zeros as fallback
                return torch.zeros_like(token_embeds)
