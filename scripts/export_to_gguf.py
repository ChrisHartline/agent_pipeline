#!/usr/bin/env python3
"""Export a full checkpoint to GGUF (stub).

Actual conversion uses llama.cpp conversion tools; this script is a stub that
will be extended to call the conversion CLI or Python bindings when available.
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to full checkpoint dir or safetensors file")
    parser.add_argument("--out", default="models/gguf", help="Output folder for gguf files")
    parser.add_argument("--quantize", action="store_true", help="Whether to quantize during conversion")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Stub: converting {args.checkpoint} -> {args.out} (quantize={args.quantize})")
    print("Implement conversion using llama.cpp or community conversion tools.")


if __name__ == "__main__":
    main()
