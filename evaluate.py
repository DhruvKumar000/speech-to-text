"""
Measure accuracy of the transcript.

Usage:
  python evaluate.py --reference reference.txt --hypothesis output.txt
"""

import argparse
import re
import jiwer


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    return re.sub(r"\s+", " ", text).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Correct text (typed by hand)")
    p.add_argument("--hypothesis", required=True, help="Model output text")
    args = p.parse_args()

    ref = normalize(open(args.reference, encoding="utf-8").read())
    hyp = normalize(open(args.hypothesis, encoding="utf-8").read())

    out = jiwer.process_words(ref, hyp)
    wer = out.wer
    print(f"Reference words : {len(ref.split())}")
    print(f"Output words    : {len(hyp.split())}")
    print(f"Substitutions   : {out.substitutions}")
    print(f"Deletions       : {out.deletions}")
    print(f"Insertions      : {out.insertions}")
    print(f"WER             : {wer:.2%}")
    print(f"Accuracy        : {max(0, 1 - wer):.2%}")


if __name__ == "__main__":
    main()
