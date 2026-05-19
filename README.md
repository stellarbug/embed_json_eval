
# embed_eval

Compares two JSON or text files using biomedical text embeddings and returns detailed similarity scores. Useful for evaluating LLM-generated structured outputs against a reference.

There are two comparison modes depending on your input:

- **JSON mode** — compares two JSON files field by field
- **Query block mode** — compares two structured search query files block by block

---

## Setup

```
pip install torch transformers numpy scipy matplotlib
```

Models download from HuggingFace on first run.

| preset | model | notes |
|---|---|---|
| `biogpt` | microsoft/biogpt | default, best for biomedical text |
| `medcpt` | ncbi/MedCPT-Query-Encoder
| `biobert` | dmis-lab/biobert-base-cased-v1.1

---

## JSON comparison

### CLI

```bash
python main.py --gt ground_truth.json --pred prediction.json
```

Prints scores to stdout and writes a detailed markdown report to `debug_report.md`.

```bash
python main.py --gt gt.json --pred pred.json --preset biobert --report-out results.md
```

### API

```python
from embed_eval_lib import score

res = score("gt.json", "pred.json", preset="medcpt", report_out="report.md")
```

`res` is a flat dict:

```python
res["overall"]            # combined leaf + list score
res["leaf_mean"]          # per-path diagonal similarity
res["list_mean"]          # list section matching score (steps, reagents, etc.)
res["structure_score"]    # key/path similarity (ignores values)
res["structure_f1"]       # F1 over exact path matches
res["values_score"]       # value similarity (ignores paths)
res["values_cosine"]      # block-level cosine over all values
res["leaf_details"]       # list of per-path dicts with gt, pred, sim
res["list_details"]       # list of per-list-key dicts with score and pairs
```

### Flags

| flag | default | |
|---|---|---|
| `--preset` | `medcpt` | model |
| `--report-out` | `debug_report.md` | report path |
| `--list-keys` | `steps,reagents,...` | which list fields get matched element-wise |
| `--no-list-matching` | off | skip list matching |
| `--batch-size` | `32` | |
| `--max-length-leaf` | `128` | token cap for leaf comparisons |
| `--max-length-list` | `256` | token cap for list item comparisons |
| `--max-rows-per-table` | `200` | rows per table in the report |

---

## Query block comparison

Designed for structured search queries with labeled blocks `O`, `A1`, `A2`, `B`. The `O` block is always skipped. `A1`, `A2`, and `B` are compared between the two files.

Each block is a list of OR/AND/.NOT-separated search clauses. Before matching, clauses are parsed and cleaned — `[Field]` annotations, quotes, and optionally boolean operators are stripped, and parenthesized sub-expressions are collapsed into single units.

### Input format

```
O
"english"[Language]
NOT "review"[Publication Type]
...

A1
"nucleoproteins"[MeSH Terms]
OR "protein interaction"[All Fields]
OR ( "protein"[All Fields] AND "complex"[All Fields] )
...

A2
"Immunoprecipitation"[MeSH Terms]
OR coimmunoprecipitation
...

B
"Epitope Mapping"[MeSH Terms]
OR "Two-Hybrid System Techniques"[MeSH Terms]
...
```

### CLI

```bash
python query_scorer.py --a file1.txt --b file2.txt
```

```bash
python query_scorer.py --a file1.txt --b file2.txt \
    --preset biogpt \
    --bool-mode all \
    --log-out results.md \
    --plot-out results.png
```

`--bool-mode` controls which boolean operators are stripped before matching:

| value | behaviour |
|---|---|
| `top` | strips AND/OR/NOT at the top level of each clause only — operators inside parenthesized sub-expressions are kept *(default)* |
| `all` | strips all AND/OR/NOT regardless of depth |
| `none` | keeps all operators, only strips `[Field]` annotations and quotes |

### API

```python
from embed_eval_lib import pick_device, load_encoder, MODEL_PRESETS, compare_blocks

device = pick_device()
hf_id, pooling = MODEL_PRESETS["biogpt"]
tok, mdl = load_encoder(hf_id, device)

res = compare_blocks("file1.txt", "file2.txt", tok, mdl, device, bool_mode="top")
```

Or use the convenience wrapper which loads the model and writes the report for you:
Note - the dictonary out of score_text() and compare_blocks() is the same. The score_text() 
function just prints stuff from the compare_blocks() function

```python
from embed_eval_lib import score_text

res = score_text("file1.txt", "file2.txt", preset=" biogpt",
                 log_out="report.md", plot_out="report.png", bool_mode="top")
```

### Result structure from compare_blocks()

```python
res["_meta"]["overall"]         # average match score across A1, A2, B
res["_meta"]["overall_euclid"]  # same but euclidean
res["_meta"]["missing_a"]       # blocks not found in file A
res["_meta"]["missing_b"]       # blocks not found in file B

res["A1"]["clause_cosine_score"]         # Hungarian match score (clause-level cosine)
res["A1"]["clause_euclid_score"]       # Hungarian match score (clause-level euclidean)
res["A1"]["block_cosine_score"]             # block-level cosine (whole block vs whole block)
res["A1"]["block_euclid_score"]      # block-level euclidean
res["A1"]["clauses_a"]          # parsed clauses from file A
res["A1"]["clauses_b"]          # parsed clauses from file B
res["A1"]["stats"]              # dict with matched, unmatched, perfect, fails counts
res["A1"]["rows"]               # raw match rows: [(clause_a, clause_b, score), ...]

# Pairs dict — easy programmatic access to each match
res["A1"]["pairs"]["Immunoprecipitation"]
# -> ["co-immunoprecipitation", 0.923, 0.814]
#     [matched clause,          cosine, euclidean]
```

The difference between the two scoring approaches:

- **Match scores** (`clause_cosine_score`, `clause_euclid_score`) — each clause in block A is matched one-to-one against the closest clause in block B using Hungarian assignment. Unmatched clauses score 0.
- **Block scores** (`block_cosine_score`, `block_euclid_score`) — the entire block is compressed into a single mean embedding and compared as one vector. No per-clause matching.

### Output files

- **Markdown report** (`--log-out`) — header with scores and clause counts, side-by-side clause tables per block, then full match tables with cosine and euclidean columns, sorted worst-first.
- **Plot** (`--plot-out`) — bar chart of match breakdown per block (matched/unmatched/perfect/zero) and score distribution histogram.
