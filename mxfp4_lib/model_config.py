"""Build model from an open-source architecture config with random weights."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForCausalLM


def build_model_from_arch(cfg: dict, vocab_size: int | None = None):
    """
    Load architecture from cfg['arch_model_id'] config.json only, then
    AutoModelForCausalLM.from_config(...) for random initialization.
    Does NOT download or load pretrained weight tensors.
    """
    arch_id = cfg["arch_model_id"]
    config = AutoConfig.from_pretrained(arch_id)
    config.use_cache = False
    if vocab_size is not None and hasattr(config, "vocab_size"):
        # Keep official vocab unless caller overrides; tokenizer length should match.
        if int(config.vocab_size) != int(vocab_size):
            print(
                f"[model_config] warning: config.vocab_size={config.vocab_size} "
                f"vs tokenizer={vocab_size}; keeping config.vocab_size"
            )
    # Cap positional embeddings to at least our training seq_len if smaller (unlikely)
    seq_len = cfg.get("data", {}).get("seq_len", 512)
    if hasattr(config, "max_position_embeddings"):
        if config.max_position_embeddings < seq_len:
            config.max_position_embeddings = seq_len

    model = AutoModelForCausalLM.from_config(config)
    return model, config


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["build_model_from_arch", "count_parameters"]
