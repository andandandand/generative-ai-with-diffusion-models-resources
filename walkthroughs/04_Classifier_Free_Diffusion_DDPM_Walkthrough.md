# Walkthrough: Classifier-Free Diffusion - Controllable Generation with DDPM

## Overview

This notebook represents a pivotal advancement in the DDPM course sequence, transitioning from **random generation** to **controllable generation**. While previous notebooks generated images without specific control, this notebook introduces **classifier-free diffusion guidance** - a breakthrough technique that enables generating images of specific categories while maintaining the quality and diversity of unconditional generation.

## Course Context & Motivation

After mastering basic DDPM theory (Notebook 02) and architectural optimizations (Notebook 03), we now tackle the crucial question: *How can we control what our diffusion model generates?* This notebook introduces conditional generation through category labels, laying the foundation for the text-to-image capabilities explored in Notebook 05 (CLIP).

### The Controllable Generation Challenge

Previous notebooks generated random images from the FashionMNIST distribution. While impressive, real-world applications require **conditional generation** - the ability to specify what type of image to generate. This notebook solves this through:

1.  **Category Conditioning**: Training the model to understand clothing categories
2.  **Classifier-Free Guidance**: A novel technique to enhance conditional generation
3.  **Dual Learning**: Training both conditional and unconditional models simultaneously

## Core Innovation: Classifier-Free Diffusion

### Theoretical Foundation

Traditional conditional diffusion relies on external classifiers to guide generation, but this approach has limitations:
- Requires training separate classifier models
- Can lead to adversarial gradients
- Limited flexibility in guidance strength

**Classifier-free guidance** solves these issues by learning both conditional and unconditional generation within a single model:

$$
\epsilon_{\text{guided}} = (1 + w) \times \epsilon_{\text{conditional}} - w \times \epsilon_{\text{unconditional}}
$$

Where:
-   `w` is the guidance weight (controls conditioning strength)
-   $\epsilon_{\text{conditional}}$ is noise prediction with category conditioning
-   $\epsilon_{\text{unconditional}}$ is noise prediction without conditioning

## Code Analysis & DDPM Integration

### 1. Enhanced U-Net Architecture with Category Conditioning

```python
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, n_feat=256, n_cfeat=10, height=16):
        # n_cfeat=10 for FashionMNIST categories
        self.contextembed = nn.Embedding(n_cfeat, n_feat)
```

**DDPM Theory Connection**: The embedding layer converts discrete category labels into continuous feature vectors that can be injected into the denoising process. This extends the DDPM formulation:

Original: $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$

Conditional: $p_\theta(x_{t-1}|x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t, c))$

### 2. Category Embedding Integration

```python
def forward(self, x, t, c=None):
    x = self.init_conv(x)
    if c is None:
        c = torch.zeros(x.shape[0], self.n_feat).to(x)
    else:
        c = self.contextembed(c).view(-1, self.n_feat)
```

**Key Innovation**: The model handles both conditional (`c` provided) and unconditional (`c=None`) modes within the same architecture. This dual capability is essential for classifier-free guidance.

### 3. Bernoulli Masking for Dual Learning

```python
# Randomly mask 10% of categories to None for unconditional training
context_mask = torch.bernoulli(torch.zeros(context.shape[0]) + 0.9)
context = context * context_mask.unsqueeze(-1)
context = context.int()
```

**DDPM Theory Connection**: During training, we randomly drop conditioning information with probability 0.1. This teaches the model to:
-   Generate specific categories when conditioning is present
-   Generate diverse samples when conditioning is absent
-   Interpolate between conditional and unconditional generation

### 4. The Classifier-Free Guidance Algorithm

```python
def sample_ddpm_context(model, noises, context, w=0.0):
    # Conditional prediction
    preds = model(noises, ts, context)

    # Unconditional prediction
    preds_uncond = model(noises, ts, torch.zeros_like(context))

    # Classifier-free guidance
    preds = (1 + w) * preds - w * preds_uncond
```

**Mathematical Interpretation**: This implements the core classifier-free guidance equation. The guidance weight `w` controls the trade-off:
-   `w = 0`: Pure conditional generation
-   `w > 0`: Enhanced conditional generation (sharper category features)
-   `w < 0`: Unconditional generation with category avoidance

## Training Process & Loss Function

### Enhanced Training Loop

```python
# Training with category conditioning
noise_pred = model(noisy_image, timestep, context)
loss = nn.functional.mse_loss(noise_pred, noise)
```

**DDPM Connection**: The loss function remains the standard noise prediction loss, but now the model must learn to predict noise conditioned on category information. The mathematical formulation becomes:

$L = \mathbb{E}_{x_0, \epsilon, t, c} \left[ ||\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, c)||^2 \right]$

### Bernoulli Masking Strategy

The key innovation is the random masking during training:

```python
# 90% conditional, 10% unconditional training
context_mask = torch.bernoulli(torch.zeros(context.shape[0]) + 0.9)
```

This creates a model that learns both:
1.  **Conditional distribution**: $p(x|c)$ - generate images of specific categories
2.  **Unconditional distribution**: $p(x)$ - generate diverse images without constraints

## Generation Process & Guidance Control

### Weighted Sampling Implementation

```python
def guided_sampling(model, category, guidance_weight=2.0):
    for i in range(timesteps):
        # Conditional prediction
        eps_cond = model(x, t, category)

        # Unconditional prediction
        eps_uncond = model(x, t, None)

        # Apply classifier-free guidance
        eps = (1 + guidance_weight) * eps_cond - guidance_weight * eps_uncond

        # Standard DDPM denoising step
        x = denoise_step(x, eps, t)
```

**Guidance Weight Effects**:
-   **w = 0**: Standard conditional generation
-   **w = 2**: Enhanced category features (recommended)
-   **w = 5**: Very sharp category features (may reduce diversity)
-   **w = -1**: Anti-guidance (avoid specific categories)

## Expected Outputs & Visual Results

### 1. Category-Specific Generation
With proper category conditioning, the model generates:
-   **T-shirts** when `context = 0`
-   **Trousers** when `context = 1`
-   **Pullovers** when `context = 2`
-   And so on for all 10 FashionMNIST categories

### 2. Guidance Weight Comparison
Visual comparison shows:
-   **Low guidance (w=0)**: Subtle category features
-   **Medium guidance (w=2)**: Clear category characteristics
-   **High guidance (w=5)**: Very sharp features, potential artifacts

### 3. Quality Metrics
The notebook demonstrates:
-   Maintained generation quality compared to unconditional models
-   Enhanced controllability over generated content
-   Smooth interpolation between conditional and unconditional generation

## Technical Innovations & Implementation Details

### 1. Efficient Category Embedding
```python
self.contextembed = nn.Embedding(n_cfeat, n_feat)
```
-   Maps discrete categories to continuous feature space
-   Integrates seamlessly with existing U-Net architecture
-   Enables gradient-based optimization

### 2. Dual-Mode Architecture
The model's ability to handle both conditional and unconditional inputs within the same forward pass is crucial for classifier-free guidance efficiency.

### 3. Guidance Weight Flexibility
The parameterized guidance weight allows real-time control over generation characteristics without retraining.

## Advanced Concepts & Extensions

### 1. Scaling to Color Images (Preparation for CLIP)
The notebook hints at scaling challenges:
-   Increased parameter requirements for RGB images
-   Computational complexity with higher resolutions
-   Memory management for larger models

### 2. Multi-Modal Conditioning
The category embedding approach generalizes to:
-   Text embeddings (as seen in Notebook 05)
-   Image embeddings for image-to-image tasks
-   Multi-modal conditioning combining text and images

### 3. Research Impact
Classifier-free guidance has become the standard approach for controllable generation, influencing:
-   Stable Diffusion models
-   DALL-E 2 and subsequent versions
-   Modern text-to-image systems

## Connection to Overall Course Narrative

### From Previous Notebooks:
-   **Notebook 01**: Established U-Net denoising foundations
-   **Notebook 02**: Introduced complete DDPM framework
-   **Notebook 03**: Optimized architecture for quality generation

### Current Contributions:
-   **Controllable Generation**: Transition from random to directed sampling
-   **Classifier-Free Guidance**: Breakthrough technique for conditional generation
-   **Dual Learning**: Efficient training of conditional and unconditional models

### Towards Next Notebooks:
-   **Notebook 05 (CLIP)**: Extends category conditioning to natural language
-   **Notebook 06 (Assessment)**: Applies all learned concepts in comprehensive evaluation

## Key Learning Outcomes

### 1. Theoretical Understanding
-   Conditional diffusion model formulation
-   Classifier-free guidance mathematical foundation
-   Trade-offs between controllability and diversity

### 2. Implementation Skills
-   Category embedding integration
-   Bernoulli masking for dual training
-   Weighted sampling algorithms

### 3. Practical Applications
-   Real-time guidance control
-   Category-specific image generation
-   Foundation for text-to-image systems

## Research Context & Historical Significance

### The Classifier-Free Revolution
This notebook implements ideas from the seminal paper "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2021), which solved major limitations of early conditional diffusion models:

1.  **Eliminated Classifier Dependency**: No need for separate classifier networks
2.  **Improved Guidance Quality**: Better trade-off between fidelity and diversity
3.  **Simplified Training**: Single model handles both conditional and unconditional generation

### Impact on Modern AI
Classifier-free guidance became the foundation for:
-   **Stable Diffusion**: Open-source text-to-image generation
-   **DALL-E 2**: Advanced text-to-image synthesis
-   **Midjourney**: Artistic image generation
-   **Modern Diffusion Pipelines**: Industry standard approach

## Computational Considerations & Performance

### Memory Requirements
-   Doubled inference computation (conditional + unconditional passes)
-   Minimal training overhead (same model, additional embedding layer)
-   Efficient gradient computation through shared parameters

### Quality vs. Speed Trade-offs
-   Higher guidance weights: Better category adherence, slower generation
-   Lower guidance weights: Faster generation, less category specificity
-   Optimal balance typically around w=2.0 for most applications

## Advanced Applications & Future Directions

### 1. Multi-Label Conditioning
Extension to multiple simultaneous categories:
```python
# Example: Generate "blue dress" with both color and garment conditioning
color_embed = self.color_embed(color_label)
garment_embed = self.garment_embed(garment_label)
combined_context = color_embed + garment_embed
```

### 2. Hierarchical Conditioning
Structured category relationships:
-   Coarse categories (clothing type)
-   Fine categories (specific style)
-   Attribute conditioning (color, pattern, size)

### 3. Interactive Generation
Real-time guidance adjustment:
-   Dynamic weight modification during generation
-   Progressive refinement of conditioning
-   User-guided iterative improvement

## Summary & Next Steps

This notebook represents a crucial milestone in the DDPM learning journey, introducing **classifier-free diffusion guidance** - a technique that revolutionized controllable generation. The key innovations include:

1.  **Category Conditioning**: Embedding discrete labels into continuous feature space
2.  **Dual Learning**: Training conditional and unconditional models simultaneously
3.  **Weighted Sampling**: Real-time control over generation characteristics
4.  **Bernoulli Masking**: Efficient strategy for learning both modes

The successful implementation of classifier-free guidance sets the stage for **Notebook 05 (CLIP)**, where category labels are replaced with natural language descriptions, enabling the full text-to-image generation capabilities that define modern AI systems.

Through this progression from basic denoising (Notebook 01) to controllable generation (Notebook 04), students gain comprehensive understanding of both the theoretical foundations and practical implementations that power today's most advanced generative AI systems.