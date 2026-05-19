#!/usr/bin/env python3

import argparse
from embed_eval_lib import (
    MODEL_PRESETS, BOOL_MODES, pick_device, load_encoder,
    compare_blocks, print_block_scores, write_block_log, plot_block_stats,
)


def build_arg_parser():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--a", required=True, help="First text file")
    ap.add_argument("--b", required=True, help="Second text file")
    ap.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="biogpt")
    ap.add_argument("--bool-mode", choices=BOOL_MODES, default="top",
                    help="Which AND/OR/NOT to strip before matching: "
                         "'top' = top-level only, 'all' = everywhere, 'none' = keep all")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--log-out", default="block_report.md")
    ap.add_argument("--plot-out", default="block_report.png")
    return ap


def main():
    args = build_arg_parser().parse_args()
    device = pick_device()
    hf_id, pooling = MODEL_PRESETS[args.preset]
    tok, mdl = load_encoder(hf_id, device)

    res = compare_blocks(
        args.a, args.b, tok, mdl, device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=pooling,
        bool_mode=args.bool_mode,
    )
    print("******************** ===== ", res["A1"].keys())
    print_block_scores(res, args.a, args.b)
    write_block_log(res, args.log_out, path_a=args.a, path_b=args.b,
                    model_name=hf_id, pooling=pooling, device=device)
    plot_block_stats(res, args.plot_out)
    print(f"\nDetailed log written to {args.log_out}")


if __name__ == "__main__":
    main()