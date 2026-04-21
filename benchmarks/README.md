# Workflows-MCP Benchmarking

This folder contains baseline benchmark runners for the current workflows-mcp
knowledge retrieval stack.

Implemented retrieval pipeline:

- pgvector cosine search
- PostgreSQL full-text search
- Reciprocal Rank Fusion (RRF)

Datasets are public benchmark datasets:

- LongMemEval (`longmemeval_s_cleaned.json`)
- LoCoMo (`locomo10.json`)

## 1. Install Dependencies

From the `workflows-mcp` directory:

```bash
uv sync --all-extras
uv pip install fastembed
```

## 2. Fetch Datasets

```bash
uv run python benchmarks/fetch_benchmark_datasets.py --output-dir benchmarks/data
```

Expected output files:

- `benchmarks/data/longmemeval_s_cleaned.json`
- `benchmarks/data/locomo10.json`

## 3. Run Workflows-MCP Baselines

### LongMemEval

```bash
uv run python benchmarks/workflows_longmemeval_bench.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --granularity session \
  --embed-model default \
  --search-limit 50 \
  --purge-prefix
```

### LoCoMo

```bash
uv run python benchmarks/workflows_locomo_bench.py \
  benchmarks/data/locomo10.json \
  --granularity session \
  --embed-model default \
  --top-k 10 \
  --purge-prefix
```

Embedding defaults:

- `--embed-model default` (implicit) maps to `sentence-transformers/all-MiniLM-L6-v2`
- Optional aliases: `bge-base`, `bge-large`, `nomic`, `mxbai`

## 4. Latest Baseline Results (2026-04-07)

These baselines were generated using the default embedding model and the commands above.

Result artifacts (local files, ignored by git):

- `benchmarks/results_workflows_lme_session_sentence-transformers-all-MiniLM-L6-v2_20260407_2144.jsonl`
- `benchmarks/results_workflows_locomo_session_sentence-transformers-all-MiniLM-L6-v2_top10_20260407_2238.json`

### LongMemEval (n=500)

| Metric | Value |
|---|---:|
| Recall@1 | 0.8020 |
| Recall@3 | 0.9160 |
| Recall@5 | 0.9400 |
| Recall@10 | 0.9820 |
| Recall@30 | 0.9960 |
| Recall@50 | 1.0000 |
| NDCG@10 | 0.8844 |
| NDCG@50 | 0.8853 |

### LoCoMo (session, top-10, n=1986)

| Metric | Value |
|---|---:|
| Average Recall | 0.6875 |
| Category 1 Recall | 0.6389 |
| Category 2 Recall | 0.7773 |
| Category 3 Recall | 0.5292 |
| Category 4 Recall | 0.6825 |
| Category 5 Recall | 0.6973 |

## 5. Run Optional Variants

Example: compare against a non-default embedding model.

```bash
uv run python benchmarks/workflows_longmemeval_bench.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --granularity session \
  --embed-model bge-base \
  --search-limit 50 \
  --purge-prefix

uv run python benchmarks/workflows_locomo_bench.py \
  benchmarks/data/locomo10.json \
  --granularity session \
  --top-k 10 \
  --embed-model bge-base \
  --purge-prefix
```

## 6. Multi-Service Memory Workload Benchmark (Issue #9)

This benchmark adds practical load coverage for large scoped-memory workloads across:

- high-cardinality topology (`wing` / `room` / `hall`)
- mixed operation traffic (`ingest` + scoped retrieval + context assembly)
- throughput and latency profiling

### Run

```bash
uv run python benchmarks/workflows_memory_multiservice_bench.py \
  --wings 8 \
  --rooms-per-wing 8 \
  --halls-per-room 4 \
  --records-per-hall 3 \
  --operations 300 \
  --ingest-ratio 0.20 \
  --scoped-query-ratio 0.55 \
  --context-query-ratio 0.25
```

The script prints a JSON summary and can optionally persist it with `--output`.

Dry-run (no DB writes, CLI sanity check):

```bash
uv run python benchmarks/workflows_memory_multiservice_bench.py --dry-run
```

### Safety / non-destructive behavior

- Writes are isolated to a unique source name under `benchmark:multiservice:<timestamp>`.
- Cleanup is automatic by default (source rows are purged at end of run).
- Use `--keep-data` only when you intentionally want to inspect written rows.

### Metrics to watch

- `throughput_ops_per_second` (overall mixed-workload throughput)
- `operation_mix.*.avg_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `max_ms` (latency profile per op)
- `operation_mix.*.ops_per_second` (per-op throughput)
- `quality_signals.scoped_query_avg_results` and `scoped_query_empty_ratio`
- `quality_signals.context_avg_tokens`, `context_p95_tokens`, `context_avg_memories`

## 7. Notes

- The workflows-mcp runners create temporary benchmark rows under source prefixes:
  - `benchmark:lme:*`
  - `benchmark:locomo:*`

- The `--purge-prefix` flag removes stale rows under the selected prefix before a run.
- Output files are timestamped under `benchmarks/` by default.
- Benchmark result artifacts are not versioned; rerun commands to regenerate locally.
- For reproducible reporting, keep `--embed-model`, `--granularity`, and `--top-k` explicit.
