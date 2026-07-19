# MXFP4 Route Compare Results

**Setup**: architecture from `HuggingFaceTB/SmolLM2-360M` via `from_config` (**random weights**, no pretrained tensors); **361.82M** params (960 hidden / 32 layers); shared `init_model`; WikiText-2; 3000 steps; GB200.

| Route | Train | Infer | WikiText-2 PPL | Final train loss |
|-------|-------|-------|----------------|------------------|
| R1 | bf16 | fp16 | **720.25** | 0.097 |
| R2 | bf16 + full MXFP4 PTQ | mxfp4 W4A4 (225 linears) | **697.83** | 0.097 |
| R3 | mxfp4 fake-quant | mxfp4 W4A4 (225 linears) | **681.09** | 0.482 |

## vs previous ~52M from-scratch run

| Route | 52M PPL | 360M-arch PPL |
|-------|---------|---------------|
| R1 | 1367.7 | 720.3 |
| R2 | 1302.5 | 697.8 |
| R3 | 350.8 | 681.1 |

## Conclusion

1. **Larger architecture lowers absolute PPL** for R1/R2 vs the 52M run.
2. **R1 ≈ R2 ≈ R3** on this run (gaps within ~6%): full MXFP4 PTQ is nearly lossless vs FP16; MXFP4 FQ training ends close to BF16 on test PPL (unlike the small-model run where R3 won largely via regularization).
3. Still from-scratch + WikiText-2 — absolute PPL remains high; focus on relative gaps.

Artifacts: see `results/` in this repository (`metrics.json`, train logs, generations).
