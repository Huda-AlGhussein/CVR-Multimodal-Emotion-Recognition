"""
models_fusion.py
─────────────────────────────────────────────────────────────────────────────
Cross-modal attention fusion model combining RoBERTa (text) and WavLM (audio).

HOW CROSS-MODAL ATTENTION FUSION WORKS (for your paper and supervisor)
────────────────────────────────────────────────────────────────────────
Late fusion simply averages the output probability vectors of two
independently trained models. The models never interact — they can never
correct each other's errors.

Cross-modal attention fusion connects the two modalities inside the model.
The key idea: let the TEXT representation ask questions of the AUDIO
representation, and vice versa.

Formally (for text→audio cross-attention):
  Queries Q = H_text  [batch, seq_text, d_text]
  Keys    K = H_audio [batch, seq_audio, d_audio]
  Values  V = H_audio [batch, seq_audio, d_audio]

  Attention weights = softmax( Q·K^T / sqrt(d_fuse) )
                      [batch, seq_text, seq_audio]

  C_text = Attention_weights · V
           [batch, seq_text, d_fuse]

This produces a text-augmented-by-audio representation C_text:
each text token now contains information about which audio frames
were most relevant to it.

After cross-attention, we pool each representation to a fixed-length
vector, concatenate them, and classify.

FULL ARCHITECTURE
──────────────────
RoBERTa encoder (LoRA frozen base)
  ↓ hidden states [batch, seq_text, 768]
  Project → [batch, seq_text, d_fuse=256]  (W_t)

WavLM encoder (LoRA frozen base)
  ↓ weighted-sum frame embeddings [batch, seq_audio, 768]
  Project → [batch, seq_audio, d_fuse=256]  (W_a)

Cross-modal attention:
  H_t' = CrossAttn(Q=H_t, K=H_a, V=H_a) → [batch, seq_text, d_fuse]

Pool:
  z_text  = mean(H_t)   [batch, d_fuse]
  z_cross = mean(H_t')  [batch, d_fuse]
  z_audio = mean(H_a)   [batch, d_fuse]

Concatenate: z = [z_text; z_cross; z_audio] → [batch, 3 × d_fuse = 768]

Classify:
  Linear(768 → 256) → LayerNorm → GELU → Dropout(0.3) → Linear(256 → 6)

WHY WE USE BOTH z_text AND z_cross
─────────────────────────────────────
z_text preserves the original text representation.
z_cross contains text features informed by audio.
Concatenating both lets the classifier decide how much to rely on the
original text vs. the audio-informed text on a per-sample basis.
This is called a residual information design.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, WavLMModel
from peft import LoraConfig, get_peft_model, TaskType

from config import cfg, NUM_CLASSES


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion of RoBERTa (text) and WavLM (audio).
    Both encoders use LoRA; the fusion components are fully trainable.
    """

    def __init__(self):
        super().__init__()

        # ── Text encoder (RoBERTa + LoRA) ─────────────────────────────────────
        text_base = RobertaModel.from_pretrained(
            cfg.text_model_name,
            output_hidden_states=False,   # we only need last_hidden_state here
        )
        text_lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules_text,
            bias="none",
        )
        self.text_encoder = get_peft_model(text_base, text_lora_config)

        # ── Audio encoder (WavLM + LoRA) ──────────────────────────────────────
        audio_base = WavLMModel.from_pretrained(
            cfg.audio_model_name,
            output_hidden_states=True,    # need all layers for weighted sum
        )
        # Freeze CNN feature extractor
        for param in audio_base.feature_extractor.parameters():
            param.requires_grad = False

        audio_lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules_audio,
            bias="none",
        )
        self.audio_encoder = get_peft_model(audio_base, audio_lora_config)

        # Learnable weighted sum for WavLM layers (same as WavLMLoRA)
        self.layer_weights = nn.Parameter(torch.ones(12))   # 12 transformer layers

        # ── Projection layers ─────────────────────────────────────────────────
        # Both modalities are projected to d_fuse=256 so cross-attention operates
        # in the same dimensional space.
        d_text  = 768             # RoBERTa-base hidden size
        d_audio = 768             # WavLM-base+ hidden size
        d_fuse  = cfg.fusion_dim  # 256

        self.text_proj  = nn.Linear(d_text, d_fuse)    # W_t: [768 → 256]
        self.audio_proj = nn.Linear(d_audio, d_fuse)   # W_a: [768 → 256]

        # ── Cross-modal attention ─────────────────────────────────────────────
        # nn.MultiheadAttention implements the standard scaled dot-product
        # attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        #
        # num_heads=4: we split d_fuse=256 into 4 attention heads of size 64.
        # Using multiple heads allows the model to attend to different audio
        # features for different aspects of each text token simultaneously.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_fuse,   # must match Q, K, V dimensions
            num_heads=4,
            dropout=0.1,
            batch_first=True,   # input format is [batch, seq, d] (not [seq, batch, d])
        )

        # ── Classification head ───────────────────────────────────────────────
        # Input: concat of [z_text, z_cross, z_audio] → 3 × d_fuse = 3 × 256 = 768
        self.classifier = nn.Sequential(
            nn.Linear(3 * d_fuse, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def _encode_text(
        self,
        input_ids:      torch.Tensor,   # [batch, seq_text]
        attention_mask: torch.Tensor,   # [batch, seq_text]
    ) -> torch.Tensor:
        """
        Encodes text and returns last-layer hidden states.
        OUTPUT: [batch, seq_text, 768]
        """
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return output.last_hidden_state   # [batch, seq_text, 768]

    def _encode_audio(
        self,
        waveform:   torch.Tensor,   # [batch, max_audio_samples]
        audio_mask: torch.Tensor,   # [batch, max_audio_samples]
    ) -> torch.Tensor:
        """
        Encodes audio and returns weighted-sum frame embeddings.
        OUTPUT: [batch, num_frames, 768]
        """
        output = self.audio_encoder(
            input_values=waveform,
            attention_mask=audio_mask,
        )
        # Weighted sum of 12 transformer layers (same as WavLMLoRA)
        transformer_states = output.hidden_states[1:]    # tuple of 12 tensors
        norm_weights = torch.softmax(self.layer_weights, dim=0)  # [12]
        stacked      = torch.stack(transformer_states, dim=0)    # [12, batch, frames, 768]
        weighted     = torch.einsum("lbfd,l->bfd", stacked, norm_weights)
        return weighted   # [batch, num_frames, 768]

    def _fuse(
        self,
        input_ids:      torch.Tensor,   # [batch, seq_text]
        attention_mask: torch.Tensor,   # [batch, seq_text]
        waveform:       torch.Tensor,   # [batch, max_audio_samples]
        audio_mask:     torch.Tensor,   # [batch, max_audio_samples]
    ) -> torch.Tensor:
        """
        Shared backbone for forward() and embed(). Runs steps 1–5 and
        returns z = [z_text; z_cross; z_audio], shape [batch, 3*d_fuse=768].
        Stops before the classifier head so embed() can return z directly.
        """
        # ── Step 1: Encode text ───────────────────────────────────────────────
        H_text  = self._encode_text(input_ids, attention_mask)   # [B, S_t, 768]

        # ── Step 2: Encode audio ──────────────────────────────────────────────
        H_audio = self._encode_audio(waveform, audio_mask)       # [B, S_a, 768]

        # ── Step 3: Project to shared d_fuse space ────────────────────────────
        H_t = self.text_proj(H_text)    # [B, S_t, 256]
        H_a = self.audio_proj(H_audio)  # [B, S_a, 256]

        # ── Step 4: Cross-modal attention (text queries audio) ────────────────
        # Q = H_t (text asks: "which audio frames are relevant to each token?")
        # K = H_a (audio provides: "here are the keys to my frames")
        # V = H_a (audio provides: "here are the values / content of my frames")
        #
        # key_padding_mask: tells attention which audio frames are padding.
        # nn.MultiheadAttention uses True=ignore, False=attend.
        # Our audio_mask uses 1=real, 0=padding → we invert it.
        num_audio_frames = H_a.shape[1]
        frame_padding_mask = self._subsample_mask(audio_mask, num_audio_frames)
        # True where padding, False where real audio
        frame_padding_mask = (frame_padding_mask == 0)  # [B, num_frames]

        # C_ta: for each text token, a weighted combination of audio frames
        C_ta, attn_weights = self.cross_attention(
            query=H_t,                            # [B, S_t, 256]
            key=H_a,                              # [B, S_a, 256]
            value=H_a,                            # [B, S_a, 256]
            key_padding_mask=frame_padding_mask,  # [B, S_a] True=ignore
        )
        # C_ta shape: [B, S_t, 256]
        # attn_weights shape: [B, S_t, S_a] — available for XAI visualisation

        # ── Step 5: Mean-pool each representation ─────────────────────────────
        text_mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, S_t, 1]
        z_text = (H_t * text_mask_expanded).sum(dim=1) / \
                  text_mask_expanded.sum(dim=1).clamp(min=1)   # [B, 256]

        z_cross = (C_ta * text_mask_expanded).sum(dim=1) / \
                   text_mask_expanded.sum(dim=1).clamp(min=1)  # [B, 256]

        frame_mask_float = (~frame_padding_mask).unsqueeze(-1).float()  # [B, S_a, 1]
        z_audio = (H_a * frame_mask_float).sum(dim=1) / \
                   frame_mask_float.sum(dim=1).clamp(min=1)    # [B, 256]

        # z: [B, 3×256] = [B, 768]
        return torch.cat([z_text, z_cross, z_audio], dim=1)

    def forward(
        self,
        input_ids:      torch.Tensor,   # [batch, seq_text]
        attention_mask: torch.Tensor,   # [batch, seq_text]
        waveform:       torch.Tensor,   # [batch, max_audio_samples]
        audio_mask:     torch.Tensor,   # [batch, max_audio_samples]
    ) -> torch.Tensor:
        """
        Forward pass: (text + audio) → class logits.
        OUTPUT: logits FloatTensor [batch_size, NUM_CLASSES]
        """
        z = self._fuse(input_ids, attention_mask, waveform, audio_mask)
        return self.classifier(z)   # [B, NUM_CLASSES]

    def embed(
        self,
        input_ids:      torch.Tensor,   # [batch, seq_text]
        attention_mask: torch.Tensor,   # [batch, seq_text]
        waveform:       torch.Tensor,   # [batch, max_audio_samples]
        audio_mask:     torch.Tensor,   # [batch, max_audio_samples]
    ) -> torch.Tensor:
        """
        Returns the penultimate fusion embedding z [batch, 768] without
        passing through the classifier head. Used by the RAG component in
        agentic_ai/ to build and query the kNN retrieval index.

        z = concat(z_text, z_cross, z_audio) — the same vector that forward()
        feeds into self.classifier. Extracting it here rather than via a
        forward hook keeps the interface explicit and readable.
        """
        # [AGENT3-EMBED] Embedding extraction point for kNN retrieval index.
        return self._fuse(input_ids, attention_mask, waveform, audio_mask)

    def _subsample_mask(
        self, audio_mask: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """
        Subsamples audio_mask from sample level to frame level.
        Identical to WavLMLoRA._compute_frame_mask().
        Returns [batch, num_frames] LongTensor.
        """
        batch_size = audio_mask.shape[0]
        frame_mask = torch.zeros(
            batch_size, num_frames, dtype=torch.long, device=audio_mask.device
        )
        stride = 320
        for f in range(num_frames):
            sample_idx = f * stride
            if sample_idx < audio_mask.shape[1]:
                frame_mask[:, f] = audio_mask[:, sample_idx]
        return frame_mask
