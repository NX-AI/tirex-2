# TiRex-2 Pro

TiRex-2 already provides state-of-the-art performance for zero-shot prediction, so this
open-source release can be used as-is without training on your own data.

A Pro version extends TiRex-2 with additional capabilities, including:

- **Streaming**: incremental forecast updates as new observations arrive, without recomputing
  over the full history. (This open-source release recomputes over the full context on every
  call — see [How-to: Streaming](how-to/streaming.md) for the exact distinction.)
- **Speed**: performance-optimized inference, including optimization for dedicated hardware
  such as edge, embedded, and industrial PC deployments.
- **Finetuning**: models fine-tuned on your data or with different pretraining.
- **Classification & Regression**: TiRex-2 adapted for classification and regression tasks.

These are Pro-exclusive additions — this documentation does not cover them as usable APIs of
the open-source package, since they aren't part of it.

If you are interested in any of these, contact [contact@nx-ai.com](mailto:contact@nx-ai.com).
