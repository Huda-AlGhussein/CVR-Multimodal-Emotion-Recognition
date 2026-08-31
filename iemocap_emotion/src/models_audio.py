"""
models_audio.py
─────────────────────────────────────────────────────────────────────────────
WavLM-base+ with LoRA fine-tuning for 6-class emotion classification.

HOW WAVLM WORKS (for your paper and supervisor)
─────────────────────────────────────────────────
WavLM (Chen et al., 2022) is a self-supervised speech model pretrained on
94,000 hours of audio (60K hours LibriLight + 10K hours GigaSpeech + 24K
hours VoxPopuli).

ARCHITECTURE (WavLM-base+):
  1. CNN Feature Extractor: 7 convolutional layers
     Input:  raw waveform [batch, num_samples]
     Output: frame-level features [batch, num_frames, 512]
     Subsampling: one output frame per 320 input samples at 16 kHz
                  → 50 frames per second
     This CNN is always FROZEN (even in full fine-tuning); LoRA is not
     applied here. It is a fixed feature extractor.

  2. Transformer Encoder: 12 layers, d_model=768
     Input:  CNN features [batch, num_frames, 512] (projected to 768)
     Output: frame-level contextual features [batch, num_frames, 768]
     This is where LoRA is applied (Q and V projections in all 12 layers).

PRETRAINING OBJECTIVE (why WavLM is best for emotion)
────────────────────────────────────────────────────────
WavLM uses masked speech prediction WITH denoising:
  - Some input frames are masked (like BERT's masked language modelling).
  - The model must predict the original clean speech even when noise is added.

This denoising objective forces WavLM to learn representations that are
robust to noise AND sensitive to the acoustic properties of speech (pitch,
energy, speaking rate) — exactly what emotion recognition needs.

Wav2Vec2 uses only contrastive masked prediction (no denoising).
Whisper is trained for ASR (transcription), making it less sensitive to
paralinguistic features. These design differences explain why WavLM
outperforms both on emotion tasks (SUPERB benchmark).

POOLING STRATEGY: WEIGHTED SUM OF ALL LAYERS
─────────────────────────────────────────────
Unlike text models where the last layer is usually best, speech models
encode different information at different layers:
  Lower layers → speaker identity, phonetics, basic acoustic patterns
  Upper layers → linguistic content (what was said)
  Middle layers → prosody, speaking style, paralinguistic cues (emotion)

The SUPERB benchmark (Yang et al., 2021) established that for emotion
recognition, using a weighted sum of ALL 12 transformer layer outputs
significantly outperforms using only the last layer.

We add 12 learnable weights (one per layer). These are normalised with
softmax so they sum to 1. The model learns which layers are most
informative for emotion classification.

ARCHITECTURE (this implementation)
─────────────────────────────────────
WavLM CNN Feature Extractor (frozen)
  ↓
WavLM Transformer Encoder × 12 (LoRA on Q, V in each layer)
  ↓
Weighted sum of 12 layer outputs (learnable weights, softmax normalised)
  → [batch, num_frames, 768]
  ↓
Mean pooling over time dimension (non-padded frames only)
  → [batch, 768]
  ↓
Linear(768 → 256) → LayerNorm → GELU → Dropout(0.3) → Linear(256 → 6)
"""

import torch
import torch.nn as nn
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model, TaskType

from config import cfg, NUM_CLASSES


class WavLMLoRA(nn.Module):
    """
    WavLM-base+ with LoRA adapters and weighted-sum layer pooling.
    """

    def __init__(self):
        super().__init__()

        # ── Step 1: Load pretrained WavLM ─────────────────────────────────────
        # output_hidden_states=True: return hidden states from all 12 layers.
        # WavLM's feature_extractor.eval() mode keeps the CNN normalisation
        # layers in eval mode even during training (standard practice).
        base_model = WavLMModel.from_pretrained(
            cfg.audio_model_name,
            output_hidden_states=True,
        )

        # Freeze the CNN feature extractor (it is not updated even in full FT).
        # The CNN extractor is base_model.feature_extractor.
        for param in base_model.feature_extractor.parameters():
            param.requires_grad = False
        # Note: LoRA will freeze the transformer layers too (except LoRA matrices).

        # ── Step 2: Apply LoRA to WavLM transformer encoder ───────────────────
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules_audio,
            # ["q_proj", "v_proj"] ← WavLM's attention projections
            # (different naming convention from BERT's "query"/"value")
            bias="none",
        )
        self.encoder = get_peft_model(base_model, lora_config)

        # ── Step 3: Learnable layer weights for weighted-sum pooling ─────────
        # WavLM has 12 transformer layers + 1 CNN output layer = 13 hidden states.
        # We use hidden_states[1:] (transformer layers only, skip CNN output).
        # n_layers = 12
        n_layers = 12
        # These 12 scalars are learnable parameters.
        # They start uniform (all equal) and the model learns which layers
        # are most informative for emotion during training.
        self.layer_weights = nn.Parameter(torch.ones(n_layers))
        # shape: [12] — one weight per transformer layer

        # ── Step 4: Classification head ───────────────────────────────────────
        # d_audio = 768 (WavLM-base+ hidden size)
        d_audio = 768

        self.classifier = nn.Sequential(
            nn.Linear(d_audio, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(
        self,
        waveform:   torch.Tensor,   # [batch, MAX_AUDIO_SAMPLES]
        audio_mask: torch.Tensor,   # [batch, MAX_AUDIO_SAMPLES]
    ) -> torch.Tensor:
        """
        Forward pass: raw waveform → class logits.

        INPUT
        ─────
        waveform:   FloatTensor [batch_size, max_audio_samples]
                    Normalised, padded raw waveform at 16 kHz.
        audio_mask: LongTensor  [batch_size, max_audio_samples]
                    1 for real audio samples, 0 for padding.

        OUTPUT
        ──────
        logits: FloatTensor [batch_size, NUM_CLASSES]
        """
        # ── Encode audio ──────────────────────────────────────────────────────
        # self.encoder() processes the raw waveform through:
        #   1. CNN feature extractor: [batch, samples] → [batch, frames, 512]
        #   2. Linear projection:                      → [batch, frames, 768]
        #   3. 12 transformer layers
        #
        # We pass attention_mask=audio_mask so WavLM knows which frames are
        # real audio vs. padding. WavLM internally subsamples the mask from
        # sample-level to frame-level (divides by 320).
        output = self.encoder(
            input_values=waveform,
            attention_mask=audio_mask,
            output_hidden_states=True,   # explicit: PEFT wrapper may not forward config flag
        )

        # output.hidden_states: tuple of 13 tensors
        #   [0]   → CNN feature extractor output, shape [batch, frames, 512]
        #   [1]   → transformer layer 1 output, shape [batch, frames, 768]
        #   ...
        #   [12]  → transformer layer 12 output, shape [batch, frames, 768]
        # We use only the 12 transformer layers: hidden_states[1:]
        transformer_hidden_states = output.hidden_states[1:]
        # transformer_hidden_states is a tuple of 12 tensors, each [batch, frames, 768]

        # ── Weighted sum of 12 layer outputs ──────────────────────────────────
        # Normalise layer weights with softmax so they sum to 1.
        # This ensures the weighted sum is a proper convex combination.
        norm_weights = torch.softmax(self.layer_weights, dim=0)
        # shape: [12] — 12 scalars that sum to 1

        # Stack all 12 hidden states into one tensor: [12, batch, frames, 768]
        stacked = torch.stack(transformer_hidden_states, dim=0)
        # shape: [12, batch, frames, 768]

        # Multiply each layer by its weight and sum over the layer dimension.
        # einsum "lbfd,l->bfd":
        #   l = layer dim (12)
        #   b = batch dim
        #   f = frames dim
        #   d = hidden dim (768)
        # Multiplies each layer by its scalar weight, then sums over l.
        weighted = torch.einsum("lbfd,l->bfd", stacked, norm_weights)
        # shape: [batch, frames, 768]

        # ── Mean pooling over frames (ignoring padding) ────────────────────────
        # WavLM's subsampled frame-level attention mask.
        # The CNN subsamples by a factor of 320 (at 16 kHz).
        # HuggingFace provides a utility to compute the subsampled mask.
        # We manually replicate it: the mask output from HuggingFace is stored
        # in output, but we compute it from scratch for clarity.
        #
        # Subsampled mask shape: [batch, num_frames]
        # We get num_frames from the actual output tensor.
        num_frames = weighted.shape[1]

        # Compute subsampled attention mask.
        # Method: sum the sample-level mask in windows of size 320, then convert
        # to binary. We use the mask from the output if available.
        if hasattr(output, "extract_features") or hasattr(output, "hidden_states"):
            # Get the frame-level mask directly from WavLM if available
            frame_mask = self._compute_frame_mask(audio_mask, num_frames)
        else:
            frame_mask = torch.ones(
                weighted.shape[0], num_frames,
                dtype=torch.long, device=weighted.device
            )

        # Expand frame_mask for broadcasting: [batch, frames] → [batch, frames, 1]
        frame_mask_expanded = frame_mask.unsqueeze(-1).float()  # [batch, frames, 1]

        # Sum weighted embeddings over non-padded frames
        sum_embeddings = (weighted * frame_mask_expanded).sum(dim=1)  # [batch, 768]

        # Count non-padded frames per batch item
        frame_counts = frame_mask_expanded.sum(dim=1).clamp(min=1)    # [batch, 1]

        # Mean = sum / count
        pooled = sum_embeddings / frame_counts   # [batch, 768]

        # ── Classify ──────────────────────────────────────────────────────────
        logits = self.classifier(pooled)  # [batch, NUM_CLASSES]
        return logits

    def _compute_frame_mask(
        self, audio_mask: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """
        Converts sample-level mask [batch, max_samples] to frame-level mask
        [batch, num_frames] by sampling every 320th position (WavLM stride).

        WHY VECTORIZED
        ───────────────
        The original implementation used a Python for-loop over num_frames
        (≈300 iterations at 6s). Each iteration executed on the CPU with a
        GPU tensor, forcing a CPU↔GPU sync per iteration. This blocked the
        forward pass. The vectorized version does the same indexing in one
        tensor operation with no Python loop and no repeated GPU syncs.
        """
        stride = 320   # WavLM CNN subsampling factor at 16 kHz
        # Build frame-index tensor on the same device as audio_mask (no CPU sync)
        frame_indices = torch.arange(num_frames, device=audio_mask.device) * stride
        # Clamp to the valid sample range (handles the last frame boundary)
        frame_indices = frame_indices.clamp(max=audio_mask.shape[1] - 1)
        # Index: audio_mask[:, frame_indices] picks one sample per frame
        return audio_mask[:, frame_indices]   # [batch, num_frames]

    def get_audio_embeddings(
        self,
        waveform:   torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns frame-level embeddings from the weighted sum of all layers.
        Used by the fusion model for cross-modal attention.

        OUTPUT
        ──────
        FloatTensor [batch_size, num_frames, 768]
        """
        output = self.encoder(
            input_values=waveform,
            attention_mask=audio_mask,
            output_hidden_states=True,
        )
        transformer_hidden_states = output.hidden_states[1:]
        norm_weights = torch.softmax(self.layer_weights, dim=0)
        stacked      = torch.stack(transformer_hidden_states, dim=0)
        weighted     = torch.einsum("lbfd,l->bfd", stacked, norm_weights)
        return weighted   # [batch, num_frames, 768]
