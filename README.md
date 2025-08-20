# language-model

## Overview
This project focuses on training language models—specifically Small Language Models (SLMs) and Masked Language Models (MLMs)—on legal data using various transformer architectures (encoder, decoder, encoder-decoder, and conditional models).

## Project Structure
- **data/**
    - `tokenizer.py`: Implements tokenization algorithms (Byte Pair Encoding, WordPiece, Unigram, SentencePiece).
    - `dataloader.py`: Handles data shuffling, batching, and sequence padding.

- **models/**
    - `embeddings.py`: Provides token and positional embeddings (sinusoidal or learned).
    - `transformer_block.py`: Contains multi-head attention, feedforward layers, normalization, and residual connections.
    - `transformer_stack.py`: Stacks multiple transformer blocks.
    - `output_projection.py`: Projects model outputs to vocabulary size.
    - `llm_model.py`: Integrates all components for the full model forward pass.

- **decoding/**
    - `decoder.py`: Implements decoding strategies (greedy, top-k, top-p/nucleus sampling).
    - `beam_search.py`: Advanced decoding methods.

- **training/**
    - `optimizer.py`: Modular optimizers (AdamW, SGD, Lion).
    - `scheduler.py`: Learning rate schedulers (cosine decay, linear warmup).
    - `trainer.py`: Training loop, loss logging, and gradient accumulation.

- **inference/**
    - `orchestrator.py`: Manages autoregressive generation and inference.

- **utils/**
    - `config.py`: Stores hyperparameters, paths, and model sizes.
    - `logging.py`: Logs training and inference statistics.

- **tests/**
    - Contains unit tests for each module to ensure cohesion and correctness.

## Future Extensions

The modular design allows for easy integration of recent LLM research:

- **Efficient Attention:** Integrate FlashAttention, Performer, or Linear Attention in `transformer_block.py`.
- **Mixture of Experts (MoE):** Replace feedforward networks with expert routing.
- **Adapters / LoRA:** Add lightweight fine-tuning modules in `llm_model.py`.
- **Speculative Decoding:** Implement alternative decoding strategies in `decoding/`.
- **KV-Cache Optimization:** Enhance `orchestrator.py` for more efficient inference.
