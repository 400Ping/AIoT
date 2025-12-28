import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
BUILD_DIRS = [CURRENT_DIR / "build", CURRENT_DIR / "build" / "Release"]
for d in BUILD_DIRS:
    sys.path.append(str(d))


def load_cuda_lib():
    try:
        import cuda_lib  # type: ignore
        return cuda_lib
    except ImportError as exc:  # pragma: no cover - runtime guard
        search_paths = ", ".join(str(p) for p in BUILD_DIRS)
        raise SystemExit(
            f"cuda_lib not found. Build it under {CURRENT_DIR}/build. "
            f"Current search paths: {search_paths}"
        ) from exc


def run_fix_code(patterns=None):
    patterns = patterns or [os.path.join(CURRENT_DIR, "src", "*.cu")]
    targets = []
    for pat in patterns:
        targets.extend(glob.glob(pat))

    if not targets:
        print("No files matched. Nothing to fix.")
        return

    for filepath in targets:
        print(f"Processing {filepath}...")
        try:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, "r", encoding="mbcs") as f:
                    content = f.read()

            new_lines = []
            if "#define NOMINMAX" not in content:
                new_lines.append("#define NOMINMAX")

            for line in content.splitlines():
                clean_line = "".join(c for c in line if ord(c) < 128)
                new_lines.append(clean_line)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
            print(" -> Fixed")
        except Exception as e:  # pragma: no cover - file system guard
            print(f" -> Error: {e}")

    print("[done] cleaned CUDA source files.")


def _run_preprocess_loop(cuda_lib, H, W, dst, warmup, count):
    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    for _ in range(warmup):
        _ = cuda_lib.preprocess(img, dst, dst)

    start = time.time()
    for _ in range(count):
        _ = cuda_lib.preprocess(img, dst, dst)
    end = time.time()

    total = end - start
    avg_ms = (total / count) * 1000
    fps = count / total if total > 0 else 0.0
    return avg_ms, fps


def run_test(count=100, warmup=1, H=1080, W=1920, dst=640):
    cuda_lib = load_cuda_lib()
    print(f"[test] input {W}x{H} -> {dst}x{dst}, warmup={warmup}, count={count}")
    avg_ms, fps = _run_preprocess_loop(cuda_lib, H, W, dst, warmup, count)
    print(f"avg: {avg_ms:.4f} ms | fps: {fps:.2f}")


def run_benchmark(count=1000, warmup=10, H=1080, W=1920, dst=640):
    cuda_lib = load_cuda_lib()
    print(f"[benchmark] input {W}x{H} -> {dst}x{dst}, warmup={warmup}, count={count}")
    avg_ms, fps = _run_preprocess_loop(cuda_lib, H, W, dst, warmup, count)
    print("=" * 40)
    print(f"avg: {avg_ms:.4f} ms")
    print(f"fps: {fps:.2f}")
    print("=" * 40)


def build_parser():
    parser = argparse.ArgumentParser(description="CUDA kernel helper tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fix = sub.add_parser("fix", help="Clean CUDA source encoding")
    p_fix.add_argument(
        "--pattern",
        action="append",
        help="Glob pattern for files to clean (default: src/*.cu)",
    )

    def add_shared(p):
        p.add_argument("--count", type=int, default=None, help="Loop count")
        p.add_argument("--warmup", type=int, default=None, help="Warm-up runs")
        p.add_argument("--width", type=int, default=1920, help="Input width")
        p.add_argument("--height", type=int, default=1080, help="Input height")
        p.add_argument("--dst", type=int, default=640, help="Output size (square)")

    p_test = sub.add_parser("test", help="Quick sanity loop for preprocess")
    add_shared(p_test)

    p_bench = sub.add_parser("benchmark", help="Longer benchmark for preprocess")
    add_shared(p_bench)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "fix":
        run_fix_code(args.pattern)
        return

    if args.cmd == "test":
        run_test(
            count=args.count or 100,
            warmup=args.warmup or 1,
            H=args.height,
            W=args.width,
            dst=args.dst,
        )
        return

    if args.cmd == "benchmark":
        run_benchmark(
            count=args.count or 1000,
            warmup=args.warmup or 10,
            H=args.height,
            W=args.width,
            dst=args.dst,
        )
        return


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
