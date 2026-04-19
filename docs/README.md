# MegaTrain-Plus docs

This directory holds the measurements and planning artifacts that back
every wall-clock claim in the top-level [README](../README.md). Organized so
a first-time reader can find the one file they want without scanning a
long flat list.

## Start here

- **[status.md](status.md)**: the single source of truth for the current
  state of MegaTrain-Plus. Read this first.
- **[suite_summary.md](suite_summary.md)**: the 12-model validation table
  (0.36B to 32.76B parameters, three architectures). 12/12 pass bit-exact
  loss match against upstream MegaTrain.
- **[progress_summary.md](progress_summary.md)**: running tally of what
  has landed, what was attempted and walked back, and what is queued.

## Deep-dives on each improvement

Under [`phases/`](phases/):

- [`phase1.md`](phases/phase1.md): backward prefetch + triple buffering
- [`phase1d.md`](phases/phase1d.md): root-cause of the DataLoader fork
  artifact that initially masqueraded as super-linear backward scaling
- [`phase2.md`](phases/phase2.md): FP8 weight transfer quantization
  (correct but currently net-negative on unified-memory hosts; documented
  with a precise microbenchmark of the CPU pack cost)
- [`phase3.md`](phases/phase3.md): **skip recompute via
  `store_all_activations`** - typically the biggest single speedup
- [`phase5.md`](phases/phase5.md): **zero-copy unflatten** - second
  biggest; saves both time and GPU memory by aliasing template params
  to the flat buffer

## Historical context

Under [`plan/`](plan/):

- [`plan.md`](plan/plan.md): the original MegaTrain-Plus plan, kept with
  a retrospective note at the top comparing predictions to measurements
- [`codex-critique.md`](plan/codex-critique.md): Codex independent
  critique of the plan. Most of its corrections are reflected in what
  actually shipped.
- [`gemini-analysis.md`](plan/gemini-analysis.md): Gemini independent
  technical analysis. Flagged the "backward forgotten loop" that became
  Phase 1A and the CPU pack overhead that Phase 2 later confirmed.

## Raw benchmark outputs

Under [`benchmarks/`](benchmarks/). Every wall-clock number cited in the
deep-dives and summary files points at a JSON here:

- `benchmarks/suite/` (24 files): per-model JSONs from the multi-model
  validation suite. Organized as `<Org>_<Model>_{baseline,plus}.json`.
- `benchmarks/final/` (6 files): end-to-end baseline vs plus A/B runs.
- `benchmarks/phase1/` (12 files): Phase 1A prefetch and Phase 1B
  num_buffers microbenchmarks, plus Phase 1C's grad worker run.
- `benchmarks/phase1d/` (7 files): DataLoader fork artifact investigation.
- `benchmarks/phase2/` (1 file): FP8 weight transfer.
- `benchmarks/phase3/` (6 files): `store_all_activations` A/B at several
  batch sizes and sequence lengths.
- `benchmarks/phase4/` (3 files): `_foreach_copy_` fusion + cached param
  lists.
- `benchmarks/phase5/` (5 files): zero-copy unflatten A/B.

### Regenerating the summary table

`suite_summary.md` is generated from the JSONs under `benchmarks/suite/`:

```bash
python scripts/merge_suite_results.py
```

Running a fresh A/B across the full suite (writes into `benchmarks/suite/`):

```bash
python scripts/benchmark_suite.py \
    --models "Qwen/Qwen2.5-7B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct" \
    --batch-size 2 --seq-len 512 --steps 5
```
