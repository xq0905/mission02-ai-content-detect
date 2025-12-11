from pathlib import Path
import argparse

def merge_one(dirs, out_dir, src_name, out_file):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / out_file
    total_lines = 0
    files_read = 0
    with output_file.open('w', encoding='utf-8') as out_f:
        for d in dirs:
            base = Path(d)
            src = base / src_name
            if src.exists() and src.is_file():
                files_read += 1
                with src.open('r', encoding='utf-8') as in_f:
                    for line in in_f:
                        if line.endswith('\n'):
                            out_f.write(line)
                        else:
                            out_f.write(line + '\n')
                        total_lines += 1
    return str(output_file), files_read, total_lines

def main():
    parser = argparse.ArgumentParser(description='Merge train.jsonl and val.jsonl from multiple directories into separate JSONL files')
    parser.add_argument('dirs', nargs='+', help='Directories containing train.jsonl and val.jsonl')
    parser.add_argument('--out-dir', default='merged_dataset', help='Output directory')
    parser.add_argument('--train-out', default='train.jsonl', help='Output filename for merged train')
    parser.add_argument('--val-out', default='val.jsonl', help='Output filename for merged val')
    args = parser.parse_args()
    train_output, train_files_read, train_total_lines = merge_one(args.dirs, args.out_dir, 'train.jsonl', args.train_out)
    val_output, val_files_read, val_total_lines = merge_one(args.dirs, args.out_dir, 'val.jsonl', args.val_out)
    print(f'TrainOutput: {train_output}')
    print(f'TrainFiles: {train_files_read}')
    print(f'TrainLines: {train_total_lines}')
    print(f'ValOutput: {val_output}')
    print(f'ValFiles: {val_files_read}')
    print(f'ValLines: {val_total_lines}')

if __name__ == '__main__':
    main()