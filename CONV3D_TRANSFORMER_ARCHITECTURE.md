# 3D Conv + Transformer Architecture

## Overview

This architecture addresses the key limitation of standard CNNs for RL: **local receptive fields cannot capture global spatial relationships and temporal dynamics.**

## Architecture Flow

```
Input: (batch, 8, 84, 84)  # 8 stacked grayscale frames
   ↓
Reshape: (batch, 1, 8, 84, 84)  # Add channel dim for 3D conv
   ↓
┌─────────────────────────────────────┐
│   3D Convolution Stem              │
│   - Processes space AND time        │
│   - Learns motion patterns          │
│   - Downsamples efficiently         │
└─────────────────────────────────────┘
   ↓
(batch, 128, 2, 9, 9)  # Spatiotemporal features
   ↓
Flatten to sequence: (batch, 162, 128)  # 2*9*9 = 162 tokens
   ↓
Add positional encoding
   ↓
┌─────────────────────────────────────┐
│   Transformer Encoder (2 layers)    │
│   - Self-attention across tokens    │
│   - Captures global relationships   │
│   - Uses SDPA (Flash Attention)     │
└─────────────────────────────────────┘
   ↓
Global Average Pooling: (batch, 128)
   ↓
┌─────────────┬─────────────┐
│Actor Head   │ Critic Head │
│(batch, 4)   │ (batch, 1)  │
└─────────────┴─────────────┘
```

## Why This Works Better

### 1. 3D Convolutions Process Spatiotemporal Features

**Standard 2D CNN:**
```python
# Treats 8 frames as 8 channels
# Cannot distinguish: "moving left" vs "static with varied lighting"
Conv2d(8, 32, kernel_size=8, stride=4)
```

**3D CNN:**
```python
# Explicitly models temporal dimension
# Kernel: (time=3, height=8, width=8)
# Can learn: "object moved 3 pixels to the left over 3 frames" = velocity!
Conv3d(1, 32, kernel_size=(3, 8, 8), stride=(1, 4, 4))
```

**What it learns:**
- Motion patterns (velocity, acceleration)
- Temporal consistency
- Object trajectories

### 2. Transformer Captures Global Context

**CNN Problem:**
```
Lander (top-left) needs to know where landing pad is (bottom-right)
With CNN: needs ~5-7 layers to build receptive field
With Transformer: one attention layer!
```

**Self-Attention Example:**
```
Token 1 (lander location)  →  Attends to  →  Token 120 (landing pad)
Token 15 (ground distance)  →  Attends to  →  Token 1 (lander)
```

**What it captures:**
- "How far am I from the landing pad?" (global spatial)
- "Am I oriented correctly?" (global orientation)
- "What's my velocity?" (temporal via 3D conv features)

### 3. Efficient Token Count

Instead of patching the full image (would be 6×6×8 = 288 tokens), we:
1. Use 3D conv to downsample to (2, 9, 9) = 162 tokens
2. Each token already contains spatiotemporal information
3. Transformer operates on meaningful features, not raw pixels

## Key Optimizations

### 1. Scaled Dot Product Attention (SDPA)

```python
nn.TransformerEncoderLayer(..., batch_first=True)
# Automatically uses torch.nn.functional.scaled_dot_product_attention
# Benefits:
#   - Fused kernel (Flash Attention v2 when available)
#   - 2-4x faster than naive attention
#   - Lower memory usage
```

### 2. Mixed Precision (bfloat16)

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    logits, value = model(x)
```

**Benefits:**
- 2x speedup on forward/backward
- 50% memory reduction
- bfloat16 > float16 (wider range, no loss scaling issues)

### 3. Gradient Scaling

```python
scaler = torch.cuda.amp.GradScaler()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Before gradient clipping!
clip_grad_norm_(params, 10.0)
scaler.step(optimizer)
```

Prevents gradient underflow with float16.

### 4. torch.compile() (Optional)

```python
model = torch.compile(model, mode="reduce-overhead")
```

- Uses TorchInductor backend
- Fuses operations
- ~20-30% speedup (varies by model)

## Model Size Comparison

```
ResNet:                 ~200K parameters
Conv3D + Transformer:   ~250K parameters

Conv3D stem:            ~150K
Transformer (2 layers): ~80K
Heads (actor/critic):   ~15K
```

Slightly larger but worth it for the architectural benefits!

## Training Tips

### 1. Warmup (Optional)

Transformers can benefit from learning rate warmup:
```python
# First 100 episodes: linearly increase LR from 0 to 1e-5
# Then constant
```

### 2. Positional Encoding Matters

We use learnable positional encoding (random init × 0.02):
```python
self.pos_embed = nn.Parameter(torch.randn(1, 162, 128) * 0.02)
```

Could also try sinusoidal encoding for better generalization.

### 3. Monitor Attention Patterns

If training plateaus, visualize what the transformer attends to:
```python
# Extract attention weights during evaluation
# See if it's attending to lander, landing pad, ground, etc.
```

## Comparison to Alternatives

| Architecture | Pros | Cons |
|-------------|------|------|
| **2D CNN (ResNet)** | Simple, well-tested | Local receptive field, treats frames as channels |
| **3D CNN only** | Spatiotemporal features | Still local receptive field, no global context |
| **Pure Transformer (ViT)** | Global context | Needs lots of data, less inductive bias |
| **3D CNN + Transformer** | Best of both! | Slightly more parameters |

## Expected Performance

With this architecture, you should see:
- Better value variance (critic differentiates states)
- Faster learning (captures relevant features better)
- More stable training (no pessimistic drift hopefully!)

The key is that the model can now answer:
- "Where am I relative to the landing pad?" → Global spatial attention
- "How fast am I moving?" → 3D conv temporal modeling
- "Should I fire engines now?" → Combined reasoning

## Further Improvements (If Needed)

1. **Recurrent connections:** Add LSTM between transformer and heads
2. **Multi-head value:** Predict value distribution instead of scalar
3. **Auxiliary tasks:** Predict next frame for representation learning
4. **Contrastive loss:** Learn better temporal representations

---

**Bottom line:** This architecture is specifically designed for the spatiotemporal reasoning required by visual RL. The 3D conv handles motion, the transformer handles global context, and the optimizations make it fast!
