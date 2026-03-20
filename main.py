#!/usr/bin/env python3

import argparse
from embed_eval_lib import (
    MODEL_PRESETS, DEFAULT_MODEL, DEFAULT_LIST_KEYS,
    pick_device, load_encoder, read_json,
    compare_json, build_structure_table, build_values_table, build_leaf_table,
    write_debug_report, print_summary, score,
)


def build_arg_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="medcpt")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length-leaf", type=int, default=128)
    ap.add_argument("--max-length-list", type=int, default=256)
    ap.add_argument("--list-keys", default=",".join(sorted(DEFAULT_LIST_KEYS)))
    ap.add_argument("--no-list-matching", action="store_true")
    ap.add_argument("--report-out", default="debug_report.md")
    ap.add_argument("--max-rows-per-table", type=int, default=200)
    return ap


def main():
    args = build_arg_parser().parse_args()
    device = pick_device()
    hf_id, pooling = MODEL_PRESETS[args.preset]
    tok, mdl = load_encoder(hf_id, device)

    gt_json = read_json(args.gt)
    pr_json = read_json(args.pred)
    list_keys = {k.strip() for k in args.list_keys.split(",") if k.strip()}

    res = compare_json(
        gt_json, pr_json, tok, mdl, device=device,
        batch_size=args.batch_size,
        max_length_leaf=args.max_length_leaf,
        max_length_list=args.max_length_list,
        list_keys=list_keys,
        do_list_matching=not args.no_list_matching,
        pooling=pooling,
    )

    structure_score, structure_rows = build_structure_table(
        gt_json, pr_json, tok, mdl, device,
        batch_size=args.batch_size, max_length_keys=64,
        pooling=pooling, normalize_paths=True,
    )
    values_score, values_rows = build_values_table(
        gt_json, pr_json, tok, mdl, device,
        batch_size=args.batch_size, max_length_values=64, pooling=pooling,
    )
    leaf_score, leaf_rows = build_leaf_table(
        gt_json, pr_json, tok, mdl, device,
        batch_size=args.batch_size, max_length_leaf=args.max_length_leaf, pooling=pooling,
    )

    write_debug_report(
        args.report_out, model_name=hf_id, pooling=pooling, device=device,
        structure_score=structure_score, structure_rows=structure_rows,
        values_score=values_score, values_rows=values_rows,
        leaf_score=leaf_score, leaf_rows=leaf_rows,
        max_rows_per_table=args.max_rows_per_table,
    )

    print(f"Report written to {args.report_out}")
    print_summary(args.model, res)


if __name__ == "__main__":
    main()
