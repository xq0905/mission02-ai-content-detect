import json
from sklearn.model_selection import train_test_split
import argparse
import os

parser = argparse.ArgumentParser(description="Process dataset for AI content detection")
parser.add_argument("--input_filepath", type=str, required=True, help="Path to input JSONL file")

args = parser.parse_args()


def process_dataset():
    
    input_file = args.input_filepath
    output_file = os.path.join(os.path.dirname(input_file), "processed_dataset.jsonl")
    
    all_results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                text = data['text']
                labels = data['labels']
                
                words = text.split()
                total_length = len(words)
                
                if total_length != len(labels):
                    print(f"警告：第{line_num}行，文本长度({total_length})与标签长度({len(labels)})不匹配")
                    continue
                
                if total_length == 0:
                    continue
                
                step_size = max(1, int(total_length * 0.1))
                
                current_end = step_size
                while current_end <= total_length:
                    window_words = words[0:current_end]
                    window_labels = labels[0:current_end]
                    
                    score = sum(window_labels) / len(window_labels)
                    
                    if 0 < score < 1:
                        score = 0.5
                    
                    window_text = " ".join(window_words)
                    
                    result = {
                        "text": window_text,
                        "score": score
                    }
                    all_results.append(result)
                    
                    current_end += step_size
                
                if current_end - step_size < total_length:
                    window_words = words[0:total_length]
                    window_labels = labels[0:total_length]
                    score = sum(window_labels) / len(window_labels)
                    if 0 < score < 1:
                        score = 0.5
                    window_text = " ".join(window_words)
                    result = {
                        "text": window_text,
                        "score": score
                    }
                    all_results.append(result)
                    
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"警告：第{line_num}行处理错误: {e}")
                continue
    
    print(f"总共生成了 {len(all_results)} 条数据")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"已保存到 {output_file}")
    
    train_data, val_data = train_test_split(all_results, test_size=0.2, random_state=42)
    
    train_file = os.path.join(os.path.dirname(output_file), "train.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    val_file = os.path.join(os.path.dirname(output_file), "val.jsonl")
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"训练集: {len(train_data)} 条数据，保存到 {train_file}")
    print(f"验证集: {len(val_data)} 条数据，保存到 {val_file}")
    
    scores = [item['score'] for item in all_results]
    score_0 = sum(1 for s in scores if s == 0)
    score_05 = sum(1 for s in scores if s == 0.5)
    score_1 = sum(1 for s in scores if s == 1)
    
    print(f"\n得分分布:")
    print(f"  score=0.0: {score_0} 条 ({score_0/len(scores)*100:.2f}%)")
    print(f"  score=0.5: {score_05} 条 ({score_05/len(scores)*100:.2f}%)")
    print(f"  score=1.0: {score_1} 条 ({score_1/len(scores)*100:.2f}%)")

if __name__ == "__main__":
    process_dataset()
