# Task 4 Report: 实现受限计算、统计、单位和来源处理

## Status

**DONE**

## Summary

Extended `BundledProcedureProvider` with five local Procedures on the same describe/invoke path as memory and web search:

| Procedure | Role |
|---|---|
| `builtin.calculate` | AST-restricted arithmetic (no `eval`) |
| `builtin.statistics` | `mean/median/stdev/pstdev/min/max/quantiles` |
| `builtin.convert_units` | Explicit factor/offset tables (length/mass/time/temperature/data) |
| `builtin.normalize_urls` | Scheme/host/IDNA/port/path normalize + dedupe |
| `builtin.organize_provenance` | Claim↔source map, unbacked claims, duplicate URLs |

## TDD Evidence

1. **RED** — Wrote `tests/procedures/test_analysis.py` and `tests/procedures/test_provenance.py` first (brief unsafe-expression + query-fidelity cases plus coverage for stats/units/dedupe/organize). Collection failed with `ModuleNotFoundError` for `analysis` / `provenance`.
2. **GREEN** — Implemented `procedures/bundled/analysis.py` and `provenance.py`; registered definitions/handlers in `provider.py`.
3. **VERIFY** — `PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest tests/procedures/test_analysis.py tests/procedures/test_provenance.py -v` → **26 passed**.

Regression (bundled provider / memory / web search + new tests): **68 passed**.

## Safety locks (calculator)

- Allowed: numeric constants, `+ - * / // % **`, unary `+/-`, parentheses via AST.
- Rejected as `unsafe_expression`: imports, attribute access, comprehensions, oversized AST (>128 nodes), `|n|>1e100`, `|exponent|>100`.
- Division by zero → `division_by_zero`.
- No `eval`, no third-party expression/unit parsers.

## URL / provenance locks

- Normalize: lowercase scheme/host, IDNA validate, strip default ports, RFC 3986 dot-segments, drop fragment; **preserve query order/values** (no silent tracking strip).
- Dedupe key = normalized URL; keep first provenance fields; merge later `source_id`s.
- `organize_provenance` does not judge truth or rewrite snippets.

## Files Changed

| Path | Action |
|---|---|
| `lunagentic_research_swarm/procedures/bundled/analysis.py` | Created |
| `lunagentic_research_swarm/procedures/bundled/provenance.py` | Created |
| `lunagentic_research_swarm/procedures/bundled/provider.py` | Merged analysis + provenance into describe/invoke |
| `tests/procedures/test_analysis.py` | Created |
| `tests/procedures/test_provenance.py` | Created |

## Commits

- `c479900` — `feat: add safe analysis and provenance procedures`

## Concerns

1. Unit aliases are a fixed explicit set (`m/km/...`, `kg/g/...`, `s/min/...`, `C/K/F`, `B/KiB/...`); uncommon aliases are `unknown_unit` by design.
2. IDNA rejection relies on `str.encode("idna")` (stdlib); exotic hosts that the codec accepts silently are not additionally rejected.
3. Task 5+ (LanceDB vectors) not started.

## Verification Command

```bash
cd /mnt/klein/work/maibot-plugins/maibot-lunagentic-research-swarm/.worktrees/lrs-runtime-reporting
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/procedures/test_analysis.py \
  tests/procedures/test_provenance.py -v
```

Result: **26 passed**.

## Review-1 P2 Fix Notes

**Findings addressed:**

1. **Expression length before parse** — `calculate()` rejects `len(expression) > 2000` as `unsafe_expression` before `ast.parse`; `RecursionError` / `MemoryError` / `ValueError` from parsing also map to `unsafe_expression`.
2. **Numeric overflow** — `OverflowError` (and similar) from `float()` conversion or arithmetic is caught and returned as structured `unsafe_expression` (e.g. `1e100 ** 100`), not raised / `provider_call_failed`.
3. **IPv6 brackets** — After IDNA, hosts containing `:` are wrapped in `[]` when rebuilding `netloc` (default-port strip and non-default ports both valid).
4. **Duplicate `claim_id`s** — `organize_provenance` rejects repeats with `invalid_arguments` so `claim_sources` and `unbacked_claims` cannot disagree.

**Evidence:**
```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/procedures/test_analysis.py \
  tests/procedures/test_provenance.py -v
```
Result: **31 passed**.

## Review-2 Important Fix Notes

**Findings addressed:**

1. **Non-finite statistics results** — After `mean`/`median`/`stdev`/`pstdev`/`min`/`max`/`quantiles`, `_stats_result_finite` rejects non-finite scalars or list members with `invalid_arguments` (e.g. `quantiles([-1e308, 1e308], n=100)`), avoiding `ProcedureResult` ValidationError → `provider_call_failed`.
2. **Unit-conversion overflow** — Factor and temperature paths check `math.isfinite(result)` before `_success`; overflow (e.g. `1e308 TB→B`, extreme `C→F`) returns structured `invalid_arguments`.
3. **Duplicate `source_id`s** — `organize_provenance` rejects repeated `source_id` with `invalid_arguments` before building `known_ids` / `source_rows`, matching the claim_id guard.

**Evidence:**
```bash
PYTHONPATH=.:../maibot-plugin-sdk .venv/bin/pytest \
  tests/procedures/test_analysis.py \
  tests/procedures/test_provenance.py -v
```
Result: **34 passed**.
