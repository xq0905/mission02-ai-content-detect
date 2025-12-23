import argparse
from pathlib import Path
import json
import sys

def find_files(input_dir: Path, pattern: str, recursive: bool):
    return list((input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)))

def merge_jsonl(input_dir: Path, output_file: Path, pattern: str, recursive: bool, validate: bool):
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"输入目录不存在或不是目录: {input_dir}", file=sys.stderr)
        sys.exit(1)
    files = [p for p in find_files(input_dir, pattern, recursive) if p.is_file()]
    files = [p for p in files if p.resolve() != output_file]
    files.sort(key=lambda p: p.name.lower())
    if not files:
        print("未找到匹配的 jsonl 文件", file=sys.stderr)
        sys.exit(1)
    total_lines = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as out:
        for p in files:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if validate:
                        try:
                            json.loads(line)
                        except Exception:
                            continue
                    out.write(line.rstrip("\n") + "\n")
                    total_lines += 1
    print(f"合并文件数: {len(files)}")
    print(f"写入行数: {total_lines}")
    print(f"输出文件: {output_file}")

def main():
    ap = argparse.ArgumentParser(prog="merge_dataset")
    ap.add_argument("--dir", dest="input_dir", default=".", help="输入目录")
    ap.add_argument("--out", dest="out", default="merged.jsonl", help="输出文件路径")
    ap.add_argument("--pattern", default="*.jsonl", help="匹配模式")
    ap.add_argument("--recursive", action="store_true", help="是否递归子目录")
    ap.add_argument("--validate", action="store_true", help="是否校验每行 JSON 格式")
    args = ap.parse_args()
    input_dir = Path(args.input_dir).resolve()
    out = Path(args.out)
    merge_jsonl(input_dir, out, args.pattern, args.recursive, args.validate)

if __name__ == "__main__":
    main()
