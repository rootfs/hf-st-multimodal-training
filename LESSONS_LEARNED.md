# Datacenter Cached Training Lessons Learned

This file captures the main findings from the May 2026 investigation into the standalone server trainer for the production datacenter tri-encoder.

## Core conclusions

- The intended production path is the mmBERT + SigLIP2 + Whisper tri-encoder, not the native Qwen example path.
- The cached dataset startup regression was real: eager shard opens at startup made the trainer appear stalled before the first step.
- Restoring metadata-driven shard sizing fixed startup latency, but did not fix steady-state GPU burstiness by itself.
- DataLoader worker prefetch is not equivalent to the old cached tensor shard-prefetch path.
- Cached shard prefetch only helps when the training loop consumes shards sequentially.

## What failed

### Random-access cached loading plus DataLoader tuning

- Increasing `num_workers` and `prefetch_factor` improved startup-to-first-step behavior.
- It did not remove bursty GPU utilization across ranks.

### Porting shard prefetch without changing access order

- A worker-local shard-prefetch port was added to the random-access cached dataset.
- Because the training loader still used global `shuffle=True`, workers requested random indices across the whole cache.
- That meant a next-shard prefetcher was solving the wrong problem, and step time did not improve.

### Sequential shards with globally shuffled shard order

- Sequential shard loading restored the correct access pattern for cached tensor prefetch.
- But globally shuffled shard order caused different ranks to start on very different shard modality regions.
- Some ranks started on image-only shards while others started on mixed image+audio shards, which increased cross-rank skew and slowed training.

## What worked

### Metadata-driven cached startup

- Use `metadata.json` to infer shard sizes whenever possible.
- Avoid opening every shard file at trainer startup just to count records.

### Sequential shard loading with aligned shard order

- Keep each rank on a different shard subset.
- Keep shard order aligned across ranks with `shuffle: false` unless a shard-balancing scheme is implemented.
- This preserves comparable modality regions across ranks while still avoiding redundant I/O.

### ETA and progress visibility

- The tri-encoder training loop now prints a tqdm progress bar with ETA and periodic step/loss logs.
- This makes it easier to distinguish true stalls from slow model initialization.

## Guidance for Hugging Face Sentence Transformers

- Use native Sentence Transformers for native multimodal checkpoints that can preprocess raw examples at training time.
- Do not try to feed the production cached tri-encoder shards directly into `SentenceTransformerTrainer`.
- The cached shards are encoder-specific tensors tied to the mmBERT + SigLIP2 + Whisper stack, not portable raw multimodal examples.
- If cached training ever needs to look more like Sentence Transformers, it still needs shard-sequential iterable loading semantics, not globally shuffled random access.

## Practical recommendation

For the production datacenter cached run in this repo:

- use `configs/train_server_datacenter_8gpu_cached.yaml`
- keep `sequential_shard_loading: true`
- keep `shuffle: false`
- treat DataLoader prefetch tuning as secondary to getting the shard access pattern right