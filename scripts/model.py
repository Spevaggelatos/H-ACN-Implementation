import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenPositionAndEmbedding(nn.Module):
    """Combined Token and Positional Embedding Layer."""
    def __init__(self, vocab_size, d_model, max_seq_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(positions)

class DepthLN(nn.Module):
    """Depth-Adaptive Layer Normalization."""
    def __init__(self, d_model, depth):
        super().__init__()
        self.depth = depth
        self.ln = nn.LayerNorm(d_model)
        self.s = nn.Parameter(torch.tensor(0.05)) # Learnable strength factor s initialized at 0.05 per layer

    def forward(self, x):
        # a = 1 + l*s
        alpha_l = 1 + (self.depth * self.s)
        return self.ln(x) * alpha_l

class CustomTransformerBlock(nn.Module):
    """
    Transformer Block with adaptive computation via Information Flow Gating.
    Supports modes: resnet, acn, h-acn, pure_damper, hybrid.
    """
    def __init__(self, d_model, num_heads, depth, mode="h-acn"):
        super().__init__()
        self.mode = mode
        self.depth = depth

        # Native PyTorch Attention for stability
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=0.1)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffw_damper_A = None
        self.ffw_damper_B = None

        if self.mode == "resnet":
            self.ffw_alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        elif self.mode == "acn":
            self.ffw_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        elif self.mode == "h-acn":
            self.ffw_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        elif self.mode == "pure_damper":
            self.ffw_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.rank = 2
            self.ffw_damper_A = nn.Parameter(torch.randn(d_model, self.rank) / (d_model ** 0.5))
            self.ffw_damper_B = nn.Parameter(torch.zeros(self.rank, d_model))
        elif self.mode == "hybrid":
            if self.depth <= 4:
                # Layers 1-4: Damper Rank 2, No Local Residual
                self.ffw_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)
                self.rank = 2
                self.ffw_damper_A = nn.Parameter(torch.randn(d_model, self.rank) / (d_model ** 0.5))
                self.ffw_damper_B = nn.Parameter(torch.zeros(self.rank, d_model))
            else:
                # Layers 5-8: Full ResNet Local Residual
                self.ffw_alpha = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x, is_causal=False):
        # ATTENTION
        norm_x = self.ln1(x)

        attn_mask = None
        if is_causal:
            seq_len = x.size(1)
            # Create a strict causal mask (upper triangular filled with -inf)
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Pass the mask and leave is_causal=False to bypass buggy backends
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, attn_mask=attn_mask, need_weights=False)

        inter_x = x + attn_out

        # FEEDFORWARD
        norm_inter = self.ln2(inter_x)
        ffn_out = self.ffn(norm_inter)

        if self.ffw_damper_A is not None:
            damper_term = (inter_x @ self.ffw_damper_A) @ self.ffw_damper_B
        else:
            damper_term = 0.0

        # Clamping of alpha in [0,1] for the local residual
        effective_alpha = torch.clamp(self.ffw_alpha, 0.0, 1.0) if self.ffw_alpha.requires_grad else self.ffw_alpha

        # Local output: f_i(x) + α * x
        layer_output = ffn_out + inter_x * effective_alpha + damper_term

        return layer_output, self.ffw_alpha

class HACN_Model(nn.Module):
    """Full Hierarchical Adaptive Computation Network (H-ACN)."""
    def __init__(self, vocab_size=8, seq_len=16, num_classes=2, d_model=128, num_heads=4, num_layers=8, mode='h-acn', task_type='classification'):
        super().__init__()
        self.mode = mode
        self.task_type = task_type
        self.vocab_size = vocab_size

        # CLS token ID = vocab_size
        self.cls_token_id = vocab_size
        # Add +1 to vocab and seq_len ONLY if we are in classification mode (for the CLS token)
        actual_vocab_size = vocab_size + 1 if task_type == 'classification' else vocab_size
        actual_seq_len = seq_len + 1 if task_type == 'classification' else seq_len

        self.embedding = TokenPositionAndEmbedding(actual_vocab_size, d_model, max_seq_len=actual_seq_len)

        self.layers = nn.ModuleList([
            CustomTransformerBlock(d_model, num_heads, depth=i+1, mode=mode) for i in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)

        if self.task_type == 'classification':
            self.classifier = nn.Linear(d_model, num_classes)
        elif self.task_type == 'ntp':
            self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.size()

        # Input the CLS token
        if self.task_type == 'classification':
            cls_tokens = torch.full((batch_size, 1), self.cls_token_id, dtype=torch.long, device=x.device)
            x = torch.cat((cls_tokens, x), dim=1)

        local_x = self.embedding(x)

        # FFW Global Path
        use_global = self.mode in ["h-acn", "acn", "hybrid","pure_damper"]
        if use_global:
            global_residual = local_x.clone()

        is_causal = (self.task_type == 'ntp')

        # List to store representations for offline probing
        hidden_states = []

        for i, layer in enumerate(self.layers):
            layer_output, alpha_raw = layer(local_x, is_causal=is_causal)

            # Correct GLOBAL ACCUMULATION: (1 - alpha_raw) * layer_output
            if use_global:
                global_residual = global_residual + layer_output * (1.0 - alpha_raw)

            local_x = layer_output

            # Save the local representation of this layer
            hidden_states.append(local_x)

        # Final computation
        final_state = torch.zeros_like(local_x)
        if use_global:
            final_state = final_state + global_residual
            if self.mode in ["h-acn", "hybrid"]:
                last_alpha = torch.clamp(self.layers[-1].ffw_alpha, 0.0, 1.0)
                final_state = final_state + local_x * last_alpha
        else:
            final_state = local_x

        final_output = self.final_ln(final_state)

        if self.task_type == 'classification':
            # FLATTEN LOGIC
            cls_repr = final_output[:, 0, :]
            main_logits = self.classifier(cls_repr)
        else:
            main_logits = self.lm_head(final_output)

        return {"main_logits": main_logits, "hidden_states": hidden_states}

    def get_alphas(self):
        """Returns the alpha values for all layers."""
        return [], [layer.ffw_alpha.item() if isinstance(layer.ffw_alpha, nn.Parameter) else layer.ffw_alpha for layer in self.layers]
