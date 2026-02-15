# Training Experiments

## Post-Training Logging (MANDATORY)

After every embedding model training run + evaluation, update `docs/retrieval/training_experiment_log.md`:

1. Add a new run section with full hyperparameters and eval results
2. Update the leaderboard table at the top
3. Include diagnosis of what worked/failed and concrete next steps

This applies to all training: MiniLM, Granite, any future model. No exceptions.

## Current Best

Granite-FT v6b: MRR=0.550 neural-only (44K clean data, 100% hard negative coverage). All new models must be compared against this.

## Key Lessons (from experiment log)

- **Data quality > data quantity**: v6b (44K clean) crushes v5 (170K noisy) — MRR 0.550 vs 0.437 (+27%)
- **Hard negative quality is critical**: 100% coverage with 8.5 avg neural-mined negatives vs 73% with 3.5 avg
- **More data of same quality doesn't help**: v7b (64K, 100% HN coverage) ≈ v6b (44K, 100% HN coverage). Three attempts (v6, v7, v7b) all failed to beat v6b. Granite-small-R2 is at its capacity ceiling for contrastive training.
- **Dev metrics can be misleading**: v6b dev NDCG peaked at 0.29 (half v5's 0.54) yet test MRR was +27% better. Training dev set != real benchmark
- **Small clean data needs more epochs**: v6b peaked at epoch 7 on 44K; v5 peaked at epoch 1.2 on 170K
- **BATCH_SAMPLER, not NO_DUPLICATES**: NO_DUPLICATES causes linearly growing step times
- **Don't change multiple variables at once**: v6 changed LR, AllNLI, max_negs, data simultaneously — impossible to diagnose failure
- **Granite needs LR 2e-5**: 5e-6 too low (v4), 3e-5 too high (v6)
