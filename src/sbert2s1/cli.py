"""Command line: answer typed-decision requests from a JSON Lines file.

    sbert2s1 --model pritamdeka/S1-PubMedBERT --input requests.jsonl --output answers.jsonl

Each input line is ``{"state": ..., "questions": {...}}`` (an optional ``"id"`` is copied through).
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sbert2s1", description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="local export directory or Hugging Face Hub id")
    ap.add_argument("--input", default="-", help="JSON Lines requests (default: stdin)")
    ap.add_argument("--output", default="-", help="JSON Lines answers (default: stdout)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--revision", default=None, help="Hub revision to pin")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--version", action="version", version=_version())
    args = ap.parse_args(argv)

    from ._inference import load
    model = load(args.model, device=args.device, revision=args.revision)
    fin = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    fout = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        rows = [json.loads(line) for line in fin if line.strip()]
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            res = model.predict_batch([(r["state"], r["questions"]) for r in chunk], args.batch_size)
            for r, out in zip(chunk, res):
                if "id" in r:
                    out = {"id": r["id"], **out}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    finally:
        if fin is not sys.stdin:
            fin.close()
        if fout is not sys.stdout:
            fout.close()
    return 0


def _version():
    from . import __version__
    return f"sbert2s1 {__version__}"


if __name__ == "__main__":
    raise SystemExit(main())
