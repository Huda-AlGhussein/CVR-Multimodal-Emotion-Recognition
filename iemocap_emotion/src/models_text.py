"""
models_text.py
─────────────────────────────────────────────────────────────────────────────
RoBERTa-base with LoRA fine-tuning for 6-class emotion classification.

HOW LORA WORKS (explained for your supervisor and paper)
──────────────────────────────────────────────────────────
Standard fine-tuning updates ALL parameters of a pretrained model.
For RoBERTa-base that means updating 125 million weights — expensive in
time, memory, and risk of overfitting on a small dataset.

LoRA (Low-Rank Adaptation; Hu et al., 2022) leaves ALL original weights
frozen and adds small trainable matrices inside the attention layers.

Specifically, for each weight matrix W (shape [d_out, d_in]) that we want
to adapt, LoRA adds:

    ΔW = B · A

where:
  A has shape [r, d_in]   ← r is the "rank" (e.g. 8)
  B has shape [d_out, r]  ← starts as all-zeros so ΔW = 0 at init

The effective output of the adapted layer becomes:
    h = W·x + (α/r) · B·A·x

where α/r is the scaling factor (α=16, r=8 → scale=2.0).

The original W is frozen. Only A and B are trained.
For RoBERTa-base: W_Q and W_V in all 12 layers = 24 weight matrices.
Each ΔW adds (r × d_in) + (d_out × r) parameters.
For d_in = d_out = 768, r = 8:
  per matrix: 8×768 + 768×8 = 12,288 parameters
  24 matrices: 24 × 12,288 ≈ 295,000 parameters
Compared to 125M total: ~0.24% are trainable. (The classification head
adds ~2M, so total trainable is ~2.4M.)

WHY THIS IS IMPORTANT FOR YOUR PAPER
──────────────────────────────────────
This is the core claim of RQ1 and H1: LoRA achieves near-full-FT
performance with ~98% fewer trainable parameters. The count_parameters()
output gives you the exact numbers for Table EX-11.

ARCHITECTURE
─────────────
RoBERTa encoder (frozen)
  ↓
[CLS] token embedding from last 4 layers concatenated
  ↓
Linear(3072 → 256) → LayerNorm → GELU → Dropout(0.3) → Linear(256 → 6)

WHY USE LAST 4 LAYERS?
───────────────────────
BERT-family models learn a hierarchy of representations:
  Lower layers → syntactic, structural features
  Upper layers → semantic, task-specific features
Concatenating the last 4 layers gives the classification head access to
both. This is the approach from the original BERT paper (Devlin et al.,
2019) and consistently outperforms using only the last layer.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
from peft import LoraConfig, get_peft_model, TaskType

from config import cfg, NUM_CLASSES


class RoBERTaLoRA(nn.Module):
    """
    RoBERTa-base with LoRA adapters and a 2-layer classification head.

    This class is a standard PyTorch nn.Module, meaning it has:
      - forward(): defines how input passes through the model
      - parameters(): returns all parameters (frozen + trainable)
      - state_dict(): returns all weights (for saving checkpoints)
    """

    def __init__(self):
        super().__init__()

        # ── Step 1: Load pretrained RoBERTa ──────────────────────────────────
        # RobertaModel loads the encoder only (no classification head).
        # output_hidden_states=True tells the model to return hidden states
        # from ALL transformer layers, not just the last one.
        # We need this to concatenate the last 4 layers.
        base_model = RobertaModel.from_pretrained(
            cfg.text_model_name,
            output_hidden_states=True,   # ← needed for last-4-layers strategy
        )

        # ── Step 2: Apply LoRA configuration ─────────────────────────────────
        # LoraConfig tells the PEFT library which layers to inject LoRA into
        # and with what rank/alpha settings.
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            # TaskType.FEATURE_EXTRACTION means we are using the model as an
            # encoder (feature extractor), not for sequence-to-sequence or
            # causal language modelling.

            r=cfg.lora_rank,         # rank of the low-rank matrices A and B
            lora_alpha=cfg.lora_alpha,  # scaling factor
            lora_dropout=cfg.lora_dropout,

            target_modules=cfg.lora_target_modules_text,
            # ["query", "value"] → LoRA is inserted into the query and value
            # projection matrices of every attention head in every layer.

            bias="none",
            # Do not add LoRA to bias terms. This is the standard setting.
        )

        # get_peft_model() wraps the base model with LoRA adapters.
        # It automatically:
        #   1. Freezes all original parameters
        #   2. Adds the A and B matrices for the target modules
        #   3. Registers only A and B as trainable
        self.encoder = get_peft_model(base_model, lora_config)
        # After this line:
        #   self.encoder.parameters()        → all parameters (frozen + LoRA)
        #   trainable parameters             → only LoRA A and B matrices

        # ── Step 3: Classification head ───────────────────────────────────────
        # Input dimension: last 4 hidden layers each have d=768,
        # concatenated → 4 × 768 = 3072.
        hidden_dim = 768 * 4   # 3072

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),    # compress 3072 → 256
            nn.LayerNorm(256),             # normalise to prevent saturation
            nn.GELU(),                     # smooth non-linearity (better than ReLU for transformers)
            nn.Dropout(cfg.fusion_dropout),# dropout for regularisation
            nn.Linear(256, NUM_CLASSES),   # final projection → 6 class logits
        )
        # The classifier is fully trainable (no LoRA; it is a new module).

    def forward(
        self,
        input_ids:      torch.Tensor,   # [batch, max_text_tokens]
        attention_mask: torch.Tensor,   # [batch, max_text_tokens]
    ) -> torch.Tensor:
        """
        Forward pass: text tokens → class logits.

        INPUT
        ─────
        input_ids:      LongTensor [batch_size, max_text_tokens]
                        Integer token IDs from the RoBERTa tokenizer.
        attention_mask: LongTensor [batch_size, max_text_tokens]
                        1 for real tokens, 0 for padding.

        OUTPUT
        ──────
        logits: FloatTensor [batch_size, NUM_CLASSES]
                Raw (un-normalised) scores for each emotion class.
                Apply softmax for probabilities; argmax for prediction.
                Do NOT apply softmax here — CrossEntropyLoss expects raw logits.
        """
        # ── Encode text ───────────────────────────────────────────────────────
        # self.encoder() runs the RoBERTa transformer.
        # output.hidden_states is a tuple of 13 tensors (1 embedding layer + 12 transformer layers),
        # each of shape [batch, seq_len, 768].
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,   # must pass at call time — PEFT wrapper
                                         # does not forward the model-config flag
        )

        # hidden_states: tuple of 13 tensors [batch, seq_len, 768]
        # hidden_states[0]  → embedding layer (not a transformer layer)
        # hidden_states[1]  → output of transformer layer 1
        # hidden_states[12] → output of transformer layer 12 (last)
        hidden_states = output.hidden_states

        # Extract the CLS token ([position 0]) from the last 4 transformer layers.
        # hidden_states[-4:] → last 4 layers (indices 9, 10, 11, 12)
        # Each has shape [batch, seq_len, 768]
        # [:, 0, :] → take position 0 (the [CLS] token) → [batch, 768]
        # torch.cat along dim=1 → [batch, 4×768] = [batch, 3072]
        cls_embeddings = torch.cat(
            [hidden_states[i][:, 0, :] for i in [-4, -3, -2, -1]],
            dim=1
        )   # shape: [batch, 3072]

        # ── Classify ──────────────────────────────────────────────────────────
        logits = self.classifier(cls_embeddings)  # [batch, NUM_CLASSES]

        return logits

    def get_text_embeddings(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the full sequence of hidden states from the last transformer
        layer (all token positions, not just CLS).

        WHY THIS EXISTS
        ────────────────
        The fusion model needs token-level embeddings for cross-modal
        attention. This method is called by the fusion model's forward pass
        to get text hidden states before the classification head.

        OUTPUT
        ──────
        FloatTensor [batch_size, seq_len, 768]
        """
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Return the last transformer layer's hidden states for all tokens.
        # output.last_hidden_state: [batch, seq_len, 768]
        return output.last_hidden_state
