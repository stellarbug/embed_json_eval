import json
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from scipy.optimize import linear_sum_assignment

MODEL_PRESETS = {
    "medcpt":  ("ncbi/MedCPT-Query-Encoder",        "cls"),
    "biobert": ("dmis-lab/biobert-base-cased-v1.1", "mean"),
    "biogpt":  ("microsoft/biogpt",                  "mean"),
}

DEFAULT_MODEL = "microsoft/biogpt"
DEFAULT_LIST_KEYS = {"steps", "reagents", "materials", "equipment", "buffers"}

NA_STRINGS = {
    "not applicable", "n/a", "na", "none", "", "-",
    "not available", "null", "n.a.", "n.a", "missing",
}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(model_name, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    if getattr(mdl.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id
    return tok, mdl


@torch.no_grad()
def embed_texts(texts, tok, mdl, device, batch_size=32, max_length=128, pooling="cls"):
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        hs = mdl(**enc).last_hidden_state

        if pooling == "cls":
            emb = hs[:, 0, :]
        elif pooling == "mean":
            mask = enc.get("attention_mask")
            if mask is None:
                emb = hs.mean(dim=1)
            else:
                mask_f = mask.unsqueeze(-1).float()
                emb = (hs * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        else:
            raise ValueError(f"Unknown pooling '{pooling}'.")

        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.vstack(all_embs)


def hungarian_match(sim):
    cost = 1.0 - sim
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j), float(sim[i, j])) for i, j in zip(r, c)]


def best_match(sim):
    return hungarian_match(sim)


def match_and_score(gt_items, pr_items, tok, mdl, device, batch_size, max_length, pooling):
    if not gt_items and not pr_items:
        return 1.0, []
    if not gt_items or not pr_items:
        rows = [(g, "<UNMATCHED>", 0.0) for g in gt_items] + \
               [("<UNMATCHED>", p, 0.0) for p in pr_items]
        return 0.0, rows
    A = embed_texts(gt_items, tok, mdl, device, batch_size, max_length, pooling)
    B = embed_texts(pr_items, tok, mdl, device, batch_size, max_length, pooling)
    pairs = best_match(A @ B.T)

    matched_gt = {i for i, _, _ in pairs}
    matched_pr = {j for _, j, _ in pairs}

    rows = [(gt_items[i], pr_items[j], float(s)) for i, j, s in pairs]
    rows += [(gt_items[i], "<UNMATCHED>", 0.0) for i in range(len(gt_items)) if i not in matched_gt]
    rows += [("<UNMATCHED>", pr_items[j], 0.0) for j in range(len(pr_items)) if j not in matched_pr]

    denom = max(len(gt_items), len(pr_items))
    score = sum(s for _, _, s in pairs) / denom if denom else 1.0
    return float(score), rows


def cosine_diag(A, B):
    return np.sum(A * B, axis=1)


def mean_cosine(texts_a, texts_b, tok, mdl, device, batch_size, max_length, pooling):
    if not texts_a and not texts_b:
        return 1.0
    if not texts_a or not texts_b:
        return 0.0
    A = embed_texts(texts_a, tok, mdl, device, batch_size, max_length, pooling)
    B = embed_texts(texts_b, tok, mdl, device, batch_size, max_length, pooling)
    a = A.mean(axis=0)
    b = B.mean(axis=0)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b))


def _strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_block(text):
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        raise ValueError("No JSON start found.")
    stack, in_str, escape = [], False, False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if escape:        escape = False
            elif ch == "\\": escape = True
            elif ch == '"':  in_str = False
            continue
        if ch == '"':    in_str = True; continue
        if ch in "{[":   stack.append(ch)
        elif ch in "}]":
            if not stack: break
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                raise ValueError("Mismatched brackets.")
            if not stack:
                return text[start : j + 1]
    raise ValueError("Could not find a complete JSON block.")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    raw = _strip_code_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_extract_first_json_block(raw))


def leaf_blower(obj, prefix=""):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            items.extend(leaf_blower(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.extend(leaf_blower(v, f"{prefix}[{i}]"))
    else:
        items.append((prefix, obj))
    return items


def collect_list_nodes(obj, prefix="", list_keys=None):
    list_keys = list_keys or DEFAULT_LIST_KEYS
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if k in list_keys and isinstance(v, list):
                found.append((p, v))
            found.extend(collect_list_nodes(v, p, list_keys))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(collect_list_nodes(v, f"{prefix}[{i}]", list_keys))
    return found


def normalize_path(p):
    return re.sub(r"\[\d+\]", "[]", p)


def get_leaf_key(path):
    parts = [x for x in re.sub(r"\[\d+\]", "", path).split(".") if x]
    return parts[-1] if parts else path


def get_parent_key(path):
    parts = [x for x in re.sub(r"\[\d+\]", "", path).split(".") if x]
    return parts[-2] if len(parts) >= 2 else ""


def value_to_text(path, value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return f"{path}: {value}"
    try:
        return f"{path}: {json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}"
    except Exception:
        return f"{path}: {value}"


def is_scalar(x):
    return x is None or isinstance(x, (str, int, float, bool))


def stable_dump(x):
    try:
        return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(x)


def smells_empty(v):
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in NA_STRINGS
    return False


def _leading_number(s):
    s = s.strip().lstrip("~≈<>≤≥")
    m = re.match(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def numeric_similarity(a, b):
    sa = str(a).strip() if a is not None else ""
    sb = str(b).strip() if b is not None else ""
    na, nb = _leading_number(sa), _leading_number(sb)
    if na is not None and nb is not None:
        return 1.0 if na == nb else 0.0
    if na is None and nb is None:
        return 1.0 if sa.lower() == sb.lower() else 0.0
    return 0.0


def combine_value_unit_siblings(leaves):
    by_parent = defaultdict(dict)
    other = {}

    for path, val in leaves.items():
        lk = get_leaf_key(path)
        par = get_parent_key(path)
        if lk in ("value", "unit") and par:
            by_parent[par][lk] = (path, val)
        else:
            other[path] = val

    result = dict(other)
    for par, kids in by_parent.items():
        if "value" in kids and "unit" in kids:
            v_str = "" if kids["value"][1] is None else str(kids["value"][1]).strip()
            u_str = "" if kids["unit"][1] is None else str(kids["unit"][1]).strip()
            result[f"{par}.__combined__"] = f"{v_str} {u_str}".strip()
        else:
            for _, (path, val) in kids.items():
                result[path] = val

    return result


def build_structure_table(gt_json, pr_json, tok, mdl, device,
                          batch_size, max_length_keys, pooling, normalize_paths=True):
    gt_leaves = dict(leaf_blower(gt_json))
    pr_leaves = dict(leaf_blower(pr_json))

    gt_paths = list(gt_leaves.keys())
    pr_paths = list(pr_leaves.keys())

    if normalize_paths:
        gt_paths = [normalize_path(p) for p in gt_paths]
        pr_paths = [normalize_path(p) for p in pr_paths]

    gt_paths = sorted(set(gt_paths))
    pr_paths = sorted(set(pr_paths))

    return match_and_score(gt_paths, pr_paths, tok, mdl, device, batch_size, max_length_keys, pooling)


def build_values_table(gt_json, pr_json, tok, mdl, device, batch_size, max_length_values, pooling):
    gt_leaves = combine_value_unit_siblings(dict(leaf_blower(gt_json)))
    pr_leaves = combine_value_unit_siblings(dict(leaf_blower(pr_json)))

    all_paths = sorted(set(gt_leaves) | set(pr_leaves))
    gt_vals, pr_vals = [], []

    for p in all_paths:
        gt_v, pr_v = gt_leaves.get(p), pr_leaves.get(p)
        if smells_empty(gt_v) and smells_empty(pr_v):
            continue
        if gt_v is not None and not smells_empty(gt_v):
            gt_vals.append(str(gt_v))
        if pr_v is not None and not smells_empty(pr_v):
            pr_vals.append(str(pr_v))

    return match_and_score(gt_vals, pr_vals, tok, mdl, device, batch_size, max_length_values, pooling)


def build_leaf_table(gt_json, pr_json, tok, mdl, device, batch_size, max_length_leaf, pooling):
    gt_leaves = dict(leaf_blower(gt_json))
    pr_leaves = dict(leaf_blower(pr_json))
    all_paths = sorted(set(gt_leaves) | set(pr_leaves))

    embed_paths, value_paths = [], []
    for p in all_paths:
        if smells_empty(gt_leaves.get(p)) and smells_empty(pr_leaves.get(p)):
            continue
        (value_paths if get_leaf_key(p) == "value" else embed_paths).append(p)

    gt_texts = [value_to_text(p, gt_leaves.get(p, "<MISSING>")) for p in embed_paths]
    pr_texts = [value_to_text(p, pr_leaves.get(p, "<MISSING>")) for p in embed_paths]

    if embed_paths:
        A = embed_texts(gt_texts, tok, mdl, device, batch_size, max_length_leaf, pooling)
        B = embed_texts(pr_texts, tok, mdl, device, batch_size, max_length_leaf, pooling)
        embed_sims = cosine_diag(A, B)
    else:
        embed_sims = np.array([], dtype=np.float32)

    embed_rows = [(gt_texts[i], pr_texts[i], float(embed_sims[i])) for i in range(len(embed_paths))]
    value_rows = [
        (
            value_to_text(p, gt_leaves.get(p, "<MISSING>")),
            value_to_text(p, pr_leaves.get(p, "<MISSING>")),
            numeric_similarity(gt_leaves.get(p), pr_leaves.get(p)),
        )
        for p in value_paths
    ]

    all_sims = list(embed_sims) + [s for _, _, s in value_rows]
    return float(np.mean(all_sims)) if all_sims else 1.0, embed_rows + value_rows


def _structure_metrics(gt_leaves, pr_leaves, tok, mdl, device, batch_size, max_length_keys, pooling):
    gt_paths = {normalize_path(p) for p in gt_leaves}
    pr_paths = {normalize_path(p) for p in pr_leaves}

    tp = len(gt_paths & pr_paths)
    fp = len(pr_paths - gt_paths)
    fn = len(gt_paths - pr_paths)

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not gt_paths else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) else 1.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    jaccard   = tp / len(gt_paths | pr_paths) if (gt_paths | pr_paths) else 1.0
    cosine    = mean_cosine(sorted(gt_paths), sorted(pr_paths),
                            tok, mdl, device, batch_size, max_length_keys, pooling)

    return dict(structure_precision=float(precision), structure_recall=float(recall),
                structure_f1=float(f1), structure_jaccard=float(jaccard),
                structure_cosine=float(cosine))


def _structure_hungarian(gt_leaves, pr_leaves, tok, mdl, device, batch_size, max_length_keys, pooling):
    gt_paths = sorted({normalize_path(p) for p in gt_leaves})
    pr_paths = sorted({normalize_path(p) for p in pr_leaves})

    if not gt_paths and not pr_paths:
        return {"structure_score": 1.0, "structure_pairs": [], "n_gt_paths": 0, "n_pr_paths": 0}
    if not gt_paths or not pr_paths:
        return {"structure_score": 0.0, "structure_pairs": [],
                "n_gt_paths": len(gt_paths), "n_pr_paths": len(pr_paths)}

    A = embed_texts(gt_paths, tok, mdl, device, batch_size, max_length_keys, pooling)
    B = embed_texts(pr_paths, tok, mdl, device, batch_size, max_length_keys, pooling)
    pairs = best_match(A @ B.T)
    denom = max(len(gt_paths), len(pr_paths))
    score = sum(s for _, _, s in pairs) / denom if denom else 1.0

    return {"structure_score": float(score), "structure_pairs": pairs[:50],
            "n_gt_paths": len(gt_paths), "n_pr_paths": len(pr_paths)}


def _values_metrics(gt_leaves, pr_leaves, tok, mdl, device, batch_size, max_length_values, pooling):
    gt_lv = combine_value_unit_siblings(gt_leaves)
    pr_lv = combine_value_unit_siblings(pr_leaves)

    gt_vals, pr_vals = [], []
    for p in sorted(set(gt_lv) | set(pr_lv)):
        gt_v, pr_v = gt_lv.get(p), pr_lv.get(p)
        if smells_empty(gt_v) and smells_empty(pr_v):
            continue
        if gt_v is not None and not smells_empty(gt_v):
            gt_vals.append(str(gt_v))
        if pr_v is not None and not smells_empty(pr_v):
            pr_vals.append(str(pr_v))

    score, pairs_sample = 1.0, []
    denom = max(len(gt_vals), len(pr_vals)) if (gt_vals or pr_vals) else 0
    if gt_vals and pr_vals:
        A = embed_texts(gt_vals, tok, mdl, device, batch_size, max_length_values, pooling)
        B = embed_texts(pr_vals, tok, mdl, device, batch_size, max_length_values, pooling)
        pairs = best_match(A @ B.T)
        score = sum(s for _, _, s in pairs) / denom if denom else 1.0
        pairs_sample = pairs[:50]

    cosine = mean_cosine(
        [v for v in gt_vals if not smells_empty(v)],
        [v for v in pr_vals if not smells_empty(v)],
        tok, mdl, device, batch_size, max_length_values, pooling,
    )
    return dict(values_score=float(score), values_cosine=float(cosine), values_pairs=pairs_sample)


def _leaf_diagonal(gt_leaves, pr_leaves, tok, mdl, device, batch_size, max_length_leaf, pooling):
    all_paths = sorted(set(gt_leaves) | set(pr_leaves))
    embed_paths, value_paths = [], []
    for p in all_paths:
        if smells_empty(gt_leaves.get(p)) and smells_empty(pr_leaves.get(p)):
            continue
        (value_paths if get_leaf_key(p) == "value" else embed_paths).append(p)

    gt_texts = [value_to_text(p, gt_leaves.get(p, "<MISSING>")) for p in embed_paths]
    pr_texts = [value_to_text(p, pr_leaves.get(p, "<MISSING>")) for p in embed_paths]

    if embed_paths:
        A = embed_texts(gt_texts, tok, mdl, device, batch_size, max_length_leaf, pooling)
        B = embed_texts(pr_texts, tok, mdl, device, batch_size, max_length_leaf, pooling)
        embed_sims = cosine_diag(A, B)
    else:
        embed_sims = np.array([], dtype=np.float32)

    details = (
        [{"path": p, "sim": float(embed_sims[i]), "gt": gt_leaves.get(p), "pred": pr_leaves.get(p)}
         for i, p in enumerate(embed_paths)] +
        [{"path": p, "sim": numeric_similarity(gt_leaves.get(p), pr_leaves.get(p)),
          "gt": gt_leaves.get(p), "pred": pr_leaves.get(p)}
         for p in value_paths]
    )
    mean_sim = float(np.mean([d["sim"] for d in details])) if details else 1.0
    return mean_sim, details


def _list_matching(gt_json, pr_json, tok, mdl, device, batch_size, max_length_list, list_keys, pooling):
    gt_lists = {p: v for p, v in collect_list_nodes(gt_json, list_keys=list_keys)}
    pr_lists = {p: v for p, v in collect_list_nodes(pr_json, list_keys=list_keys)}

    results = []
    for lp in sorted(set(gt_lists) | set(pr_lists)):
        gl, pl = gt_lists.get(lp, []), pr_lists.get(lp, [])
        if not gl and not pl:
            continue
        if not gl or not pl:
            results.append({"path": lp, "score": 0.0, "gt_len": len(gl), "pred_len": len(pl), "pairs": []})
            continue

        gt_texts = [stable_dump(x) if not is_scalar(x) else str(x) for x in gl]
        pr_texts = [stable_dump(x) if not is_scalar(x) else str(x) for x in pl]
        A = embed_texts(gt_texts, tok, mdl, device, batch_size, max_length_list, pooling)
        B = embed_texts(pr_texts, tok, mdl, device, batch_size, max_length_list, pooling)
        pairs = best_match(A @ B.T)
        denom = max(len(gl), len(pl))
        score = sum(s for _, _, s in pairs) / denom if denom else 1.0
        results.append({"path": lp, "score": float(score),
                         "gt_len": len(gl), "pred_len": len(pl), "pairs": pairs[:50]})
    return results


def compare_json(gt_json, pr_json, tok, mdl, device, batch_size=32,
                 max_length_leaf=128, max_length_list=256,
                 list_keys=None, do_list_matching=True, pooling="cls"):
    list_keys = list_keys or DEFAULT_LIST_KEYS
    gt_leaves = dict(leaf_blower(gt_json))
    pr_leaves = dict(leaf_blower(pr_json))

    leaf_mean, leaf_details = _leaf_diagonal(
        gt_leaves, pr_leaves, tok, mdl, device, batch_size, max_length_leaf, pooling)

    list_details = (_list_matching(gt_json, pr_json, tok, mdl, device,
                                   batch_size, max_length_list, list_keys, pooling)
                    if do_list_matching else [])

    list_mean = float(np.mean([x["score"] for x in list_details])) if list_details else None
    overall = float((leaf_mean + list_mean) / 2.0) if list_mean is not None else leaf_mean

    return {
        "overall": overall,
        "leaf_mean": leaf_mean,
        "list_mean": list_mean,
        "leaf_details": leaf_details,
        "list_details": list_details,
        "device": device,
        **_structure_metrics(gt_leaves, pr_leaves, tok, mdl, device, batch_size, 64, pooling),
        **_values_metrics(gt_leaves, pr_leaves, tok, mdl, device, batch_size, 64, pooling),
        **_structure_hungarian(gt_leaves, pr_leaves, tok, mdl, device, batch_size, 64, pooling),
    }


def _md_escape(x):
    s = "" if x is None else str(x)
    return s.replace("|", r"\|").replace("\n", "<br>")


def write_debug_report(out_path, model_name, pooling, device,
                       structure_score, structure_rows,
                       values_score, values_rows,
                       leaf_score, leaf_rows,
                       max_rows_per_table=200):
    def write_table(f, title, rows):
        f.write(f"\n## {title}\n\n| GT | Pred | score |\n|---|---|---:|\n")
        sorted_rows = sorted(rows, key=lambda x: x[2])
        if max_rows_per_table:
            sorted_rows = sorted_rows[:max_rows_per_table]
        for g, p, s in sorted_rows:
            f.write(f"| {_md_escape(g)} | {_md_escape(p)} | {s:.4f} |\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# JSON Comparison Debug Report\n\n")
        f.write(f"- Model: `{model_name}`\n- Pooling: `{pooling}`\n- Device: `{device}`\n\n")
        f.write(f"**Structure Hungarian score (keys only):** {structure_score:.4f}\n\n")
        f.write(f"**Values Hungarian score (values only):** {values_score:.4f}\n\n")
        f.write(f"**Leaf mean similarity (path:value):** {leaf_score:.4f}\n\n")
        write_table(f, "Table 1 — Structure-only (keys/paths)", structure_rows)
        write_table(f, "Table 2 — Values-only", values_rows)
        write_table(f, "Table 3 — Leaf comparison (original)", leaf_rows)


def print_summary(model_name, res):
    print(f"\nModel: {model_name}  |  Device: {res['device']}")
    print(f"Leaf mean:  {res['leaf_mean']:.4f}")
    if res["list_mean"] is not None:
        print(f"List mean:  {res['list_mean']:.4f}")
    print(f"Overall:    {res['overall']:.4f}")
    print(f"Structure Hungarian:  {res['structure_score']:.4f}")
    print(f"Values Hungarian:     {res['values_score']:.4f}")
    print(f"Structure cosine:     {res['structure_cosine']:.4f}")
    print(f"Values cosine:        {res['values_cosine']:.4f}")


def score(gt_path, pred_path, preset="biogpt", report_out="report.md"):
    device = pick_device()
    hf_id, pooling = MODEL_PRESETS[preset]
    tok, mdl = load_encoder(hf_id, device)

    gt_json = read_json(gt_path)
    pr_json = read_json(pred_path)
    res = compare_json(gt_json, pr_json, tok, mdl, device=device, pooling=pooling)

    structure_score, structure_rows = build_structure_table(
        gt_json, pr_json, tok, mdl, device, batch_size=32, max_length_keys=64, pooling=pooling)
    values_score, values_rows = build_values_table(
        gt_json, pr_json, tok, mdl, device, batch_size=32, max_length_values=64, pooling=pooling)
    leaf_score, leaf_rows = build_leaf_table(
        gt_json, pr_json, tok, mdl, device, batch_size=32, max_length_leaf=128, pooling=pooling)

    write_debug_report(
        report_out, model_name=hf_id, pooling=pooling, device=device,
        structure_score=structure_score, structure_rows=structure_rows,
        values_score=values_score, values_rows=values_rows,
        leaf_score=leaf_score, leaf_rows=leaf_rows,
    )

    print_summary(hf_id, res)
    return res



# ---------------------------------------------------------------------------
# Text block comparison
# ---------------------------------------------------------------------------
 
COMPARE_BLOCKS = ("A1", "A2", "B")
 
 
def parse_blocks(path):
    blocks = {}
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line in ("O", "A1", "A2", "B"):
                current = line
                blocks[current] = []
            elif current is not None:
                blocks[current].append(line)
    return blocks
 
 
BOOL_MODES = ("all", "top", "none")
 
 
def _strip_booleans_at_depth(text, target_depth):
    tokens = re.findall(r'\(|\)|[^\s()]+', text)
    result = []
    depth = 0
    for tok in tokens:
        if tok == '(':
            result.append(tok)
            depth += 1
        elif tok == ')':
            depth -= 1
            result.append(tok)
        elif tok in ('AND', 'OR', 'NOT') and depth == target_depth:
            pass
        else:
            result.append(tok)
    return ' '.join(result)
 
 
def clean_clause(text, bool_mode="top"):
    text = re.sub(r'["\']', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)

    if bool_mode == "all":
        text = re.sub(r'\b(AND|OR|NOT)\b', '', text)
    elif bool_mode == "top":
        text = _strip_booleans_at_depth(text, target_depth=0)
 
    text = re.sub(r'[()]', '', text)          # always collapse parens
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
 
def parse_clauses(lines, bool_mode="top"):
    text = re.sub(r'\s+', ' ', ' '.join(lines)).strip()
    tokens = re.findall(r'\(|\)|[^\s()]+', text)
 
    clauses = []
    buf = []
    depth = 0
 
    for tok in tokens:
        if tok == '(':
            depth += 1
            buf.append(tok)
        elif tok == ')':
            depth -= 1
            buf.append(tok)
        elif tok == 'OR' and depth == 0:
            clause = clean_clause(' '.join(buf), bool_mode)
            if clause:
                clauses.append(clause)
            buf = []
        else:
            buf.append(tok)
 
    if buf:
        clause = clean_clause(' '.join(buf), bool_mode)
        if clause:
            clauses.append(clause)
 
    return clauses
 
 
def euclid_norm(dist):
    return 1.0 / (1.0 + dist)

def euclid_from_cosine(cosine_score):
    dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - cosine_score))))
    return euclid_norm(dist)

def mean_euclidean(texts_a, texts_b, tok, mdl, device, batch_size, max_length, pooling):
    if not texts_a and not texts_b:
        return 1.0
    if not texts_a or not texts_b:
        return 0.0
    A = embed_texts(texts_a, tok, mdl, device, batch_size, max_length, pooling)
    B = embed_texts(texts_b, tok, mdl, device, batch_size, max_length, pooling)
    dist = float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))
    return euclid_norm(dist)

def _build_pairs_dict(rows):
    pairs, seen = {}, {}
    for a, b, s in rows:
        key = a
        if key in seen:
            seen[key] += 1
            key = f"{a} [{seen[a]}]"
        else:
            seen[a] = 0
        pairs[key] = [b, round(s, 6), round(euclid_from_cosine(s), 6)]
    return pairs

def block_stats(rows):
    perfect    = sum(1 for _, _, s in rows if s >= 0.9999)
    fails      = sum(1 for a, b, s in rows if s < 0.0001 and a != "<UNMATCHED>" and b != "<UNMATCHED>")
    unmatched  = sum(1 for a, b, _ in rows if a == "<UNMATCHED>" or b == "<UNMATCHED>")
    matched    = len(rows) - unmatched
    scores     = [s for a, b, s in rows if a != "<UNMATCHED>" and b != "<UNMATCHED>"]
    mean_score = float(np.mean(scores)) if scores else 0.0
    return {
        "total":      len(rows),
        "matched":    matched,
        "unmatched":  unmatched,
        "perfect":    perfect,
        "fails":      fails,
        "mean_score": mean_score,
    }
 
 
def plot_block_stats(res, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return
 
    blocks = [b for b in res if b != "_meta"]
    stats  = {b: res[b]["stats"] for b in blocks}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x, width = np.arange(len(blocks)), 0.2
    ax = axes[0]
    ax.bar(x - 1.5*width, [stats[b]["matched"]   for b in blocks], width, label="matched")
    ax.bar(x - 0.5*width, [stats[b]["unmatched"] for b in blocks], width, label="unmatched")
    ax.bar(x + 0.5*width, [stats[b]["perfect"]   for b in blocks], width, label="perfect (≥0.99)")
    ax.bar(x + 1.5*width, [stats[b]["fails"]     for b in blocks], width, label="zero score")
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_ylabel("clause count")
    ax.set_title("Match breakdown per block")
    ax.legend()

    ax2 = axes[1]
    for b in blocks:
        scores = [s for a, bv, s in res[b]["rows"]
                  if a != "<UNMATCHED>" and bv != "<UNMATCHED>"]
        if scores:
            ax2.hist(scores, bins=20, alpha=0.6, label=b)
    ax2.set_xlabel("similarity score")
    ax2.set_ylabel("count")
    ax2.set_title("Score distribution per block")
    ax2.legend()
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")
 
 
def compare_blocks(path_a, path_b, tok, mdl, device,
                   batch_size=32, max_length=128, pooling="cls", bool_mode="top"):
    blocks_a = parse_blocks(path_a)
    blocks_b = parse_blocks(path_b)

    missing_a = [b for b in COMPARE_BLOCKS if b not in blocks_a]
    missing_b = [b for b in COMPARE_BLOCKS if b not in blocks_b]

    res, scores = {}, []

    for block in COMPARE_BLOCKS:
        clauses_a = parse_clauses(blocks_a.get(block, []), bool_mode)
        clauses_b = parse_clauses(blocks_b.get(block, []), bool_mode)

        clause_score, rows = match_and_score(
            clauses_a, clauses_b, tok, mdl, device, batch_size, max_length, pooling)

        cosine_block = mean_cosine(
            clauses_a, clauses_b, tok, mdl, device, batch_size, max_length, pooling)
        euclid_block = mean_euclidean(
            clauses_a, clauses_b, tok, mdl, device, batch_size, max_length, pooling)
        euclid_match = float(np.mean([euclid_from_cosine(s) for _, _, s in rows])) if rows else 0.0

        res[block] = {
            "pairs":         _build_pairs_dict(rows),
            "rows":          rows,
            "clauses_a":     clauses_a,
            "clauses_b":     clauses_b,
            "clause_cosine_score":    clause_score,
            "clause_euclid_score":  euclid_match,
            "block_cosine_score":        cosine_block,
            "block_euclid_score": euclid_block,
            "stats":         block_stats(rows),
        }
        scores.append(clause_score)

    overall        = float(np.mean(scores)) if scores else 0.0
    overall_euclid = float(np.mean([res[b]["clause_euclid_score"] for b in COMPARE_BLOCKS if b in res]))

    res["_meta"] = {
        "overall":        overall,
        "overall_euclid": overall_euclid,
        "missing_a":      missing_a,
        "missing_b":      missing_b,
    }
    return res
 
def print_block_scores(res, path_a, path_b):
    meta = res["_meta"]
    if meta["missing_a"]:
        print(f"WARNING: {path_a} is missing blocks: {', '.join(meta['missing_a'])}")
    if meta["missing_b"]:
        print(f"WARNING: {path_b} is missing blocks: {', '.join(meta['missing_b'])}")
    if not meta["missing_a"] and not meta["missing_b"]:
        print("All blocks found in both files.")

    print(f"\nOverall cosine score:    {meta['overall']:.4f}")
    print(f"Overall euclidean score: {meta['overall_euclid']:.4f}")
    print()
    for block in COMPARE_BLOCKS:
        if block not in res:
            continue
        d, st = res[block], res[block]["stats"]
        print(f"  {block}  match_cosine: {d['clause_cosine_score']:.4f}  match_euclid: {d['clause_euclid_score']:.4f}"
              f"  block_cosine: {d['block_cosine_score']:.4f}  block_euclid: {d['block_euclid_score']:.4f}"
              f"  |  clauses A: {len(d['clauses_a'])}  B: {len(d['clauses_b'])}"
              f"  matched: {st['matched']}  unmatched: {st['unmatched']}"
              f"  perfect: {st['perfect']}  zero: {st['fails']}")
 

def write_block_log(res, path, path_a="", path_b="", model_name="", pooling="", device=""):
    def md_escape(x):
        return str(x).replace("|", r"\|").replace("\n", "<br>")

    meta = res["_meta"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Text Block Comparison Report\n\n")
        if path_a:     f.write(f"- File A: `{path_a}`\n")
        if path_b:     f.write(f"- File B: `{path_b}`\n")
        if model_name: f.write(f"- Model: `{model_name}`\n")
        if pooling:    f.write(f"- Pooling: `{pooling}`\n")
        if device:     f.write(f"- Device: `{device}`\n")
        f.write("\n")

        if meta["missing_a"]:
            f.write(f"**WARNING:** File A is missing blocks: {', '.join(meta['missing_a'])}\n\n")
        if meta["missing_b"]:
            f.write(f"**WARNING:** File B is missing blocks: {', '.join(meta['missing_b'])}\n\n")

        f.write(f"**Overall cosine score:** {meta['overall']:.4f}  ")
        f.write(f"**Overall euclidean score:** {meta['overall_euclid']:.4f}\n\n")

        for block in COMPARE_BLOCKS:
            if block not in res:
                continue
            d, st = res[block], res[block]["stats"]
            f.write(f"**{block}** — "
                    f"clauses A: {len(d['clauses_a'])} / B: {len(d['clauses_b'])} | "
                    f"match cosine: {d['clause_cosine_score']:.4f} | match euclid: {d['clause_euclid_score']:.4f} | "
                    f"block cosine: {d['block_cosine_score']:.4f} | block euclid: {d['block_euclid_score']:.4f} | "
                    f"matched: {st['matched']} | unmatched: {st['unmatched']} | "
                    f"perfect: {st['perfect']} | zero: {st['fails']}\n\n")

        for block in COMPARE_BLOCKS:
            if block not in res:
                continue
            d = res[block]
            f.write(f"\n## Block {block} — Clauses\n\n")
            f.write(f"| # | File A ({len(d['clauses_a'])} clauses) | File B ({len(d['clauses_b'])} clauses) |\n")
            f.write("|---|---|---|\n")
            for i in range(max(len(d["clauses_a"]), len(d["clauses_b"]))):
                ca = md_escape(d["clauses_a"][i]) if i < len(d["clauses_a"]) else ""
                cb = md_escape(d["clauses_b"][i]) if i < len(d["clauses_b"]) else ""
                f.write(f"| {i+1} | {ca} | {cb} |\n")

        for block in COMPARE_BLOCKS:
            if block not in res:
                continue
            f.write(f"\n## Block {block} — Matches\n\n")
            f.write("| File A | File B | cosine | euclidean |\n|---|---|---:|---:|\n")
            for a, b, s in sorted(res[block]["rows"], key=lambda x: x[2]):
                f.write(f"| {md_escape(a)} | {md_escape(b)} | {s:.4f} | {euclid_from_cosine(s):.4f} |\n")


def score_text(path_a, path_b, preset="biogpt", log_out="block_report.md", plot_out="block_report.png", bool_mode="top"):
    device = pick_device()
    hf_id, pooling = MODEL_PRESETS[preset]
    tok, mdl = load_encoder(hf_id, device)
 
    res = compare_blocks(path_a, path_b, tok, mdl, device, pooling=pooling, bool_mode=bool_mode)
    print_block_scores(res, path_a, path_b)
    write_block_log(res, log_out, path_a=path_a, path_b=path_b,
                    model_name=hf_id, pooling=pooling, device=device)
    plot_block_stats(res, plot_out)
    print(f"\nDetailed log written to {log_out}")
    return res