# FastQwen3

This repository contains a custom implementation of the Qwen3 language model, optimized for high-performance inference. It leverages custom CUDA kernels and FlashAttention to achieve significant speedups over the baseline Hugging Face implementation.

## Overview

FastQwen3 is designed for speed and efficiency. By replacing standard PyTorch modules with custom-written CUDA kernels for operations like RMSNorm and RoPE, and by integrating FlashAttention for the attention mechanism, this implementation minimizes GPU memory bandwidth bottlenecks and maximizes throughput.

## Features

- **Optimized for Inference:** The model is designed for fast autoregressive generation with a KV cache.
- **Custom CUDA Kernels:** RMSNorm and RoPE are implemented in CUDA for maximum performance.
- **FlashAttention Integration:** Uses FlashAttention for a fast and memory-efficient attention mechanism.
- **Grouped Query Attention (GQA):** Implements GQA for efficient attention computation.
- **FP16/BF16 Support:** Supports both float16 and bfloat16 data types for mixed-precision training and inference.

## Performance

FastQwen3 provides significant performance improvements over the standard PyTorch implementation. The custom CUDA kernels and FlashAttention integration lead to higher throughput and lower latency, especially for large batch sizes and long sequences.

## Benchmarks

Here are some performance benchmarks for the FastQwen2 model.

### Performance Analysis
![Performance Analysis](assets/fastqwen2_perfanal.png)

### Scaling
![Scaling](assets/fastqwen2_scaling.png)

## Installation

To install the necessary dependencies and build the custom CUDA kernels, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fastqwen3.git
    cd fastqwen3
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file based on the imports in the project.)*

3.  **Build the custom CUDA kernels:**
    ```bash
    python setup.py install
    ```

## Usage

The main model class is `FastQwen3` in `fast_qwen/arch/qwen_fast_cuda.py`. You can use it as follows:

```python
import torch
from fast_qwen.arch.qwen_fast_cuda import FastQwen3
from fast_qwen.config import QwenConfig_float16
from fast_qwen.load_weights import load_weights_fastqwen
from fast_qwen.arch.qwen_token import Qwen3Tokenizer

# Configuration
config = QwenConfig_float16()
device = torch.device("cuda")

# Model Initialization
model = FastQwen3(config).to(device)

# Load weights (replace with your weight loading logic)
# weights_dict = ...
# load_weights_fastqwen(model, config, weights_dict)

# Tokenizer
tokenizer = Qwen3Tokenizer(tokenizer_file_path="path/to/your/tokenizer.json")

# Generation
prompt = "Hello, world!"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
generated_ids = model.generate(input_ids, max_new_tokens=100)
generated_text = tokenizer.decode(generated_ids[0])

print(generated_text)
```

## Model Architecture

The model architecture is based on the Qwen3 model, with the following key components:

-   **Embedding Layer:** `nn.Embedding` for token embeddings.
-   **Transformer Blocks:** A series of Transformer blocks, each containing:
    -   **RMSNorm:** A custom CUDA implementation of RMSNorm.
    -   **Grouped Query Attention (GQA):** An attention mechanism using FlashAttention and custom CUDA RoPE.
    -   **Feed-Forward Network (FFN):** A standard FFN with SiLU activation.
-   **Final RMSNorm:** A final RMSNorm layer before the output head.
-   **Output Head:** A linear layer to project the hidden states to the vocabulary size.

## Future Work

-   [ ] Add support for more model sizes.
-   [ ] Implement quantization (e.g., GPTQ, AWQ) for further optimization.
-   [ ] Add more comprehensive benchmark scripts.
-   [ ] Release pre-compiled CUDA kernels.
