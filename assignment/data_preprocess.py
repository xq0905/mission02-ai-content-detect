import json
from sklearn.model_selection import train_test_split


def process_dataset():
    """处理数据集，使用滑动窗口生成新的数据集"""
    
    # 读取原始数据
    input_file = "data/dataset_9000.jsonl"
    output_file = "data/processed_dataset.jsonl"
    
    all_results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                text = data['text']
                labels = data['labels']
                
                # 分词
                words = text.split()
                total_length = len(words)
                
                # 确保words长度与labels长度一致
                if total_length != len(labels):
                    print(f"警告：第{line_num}行，文本长度({total_length})与标签长度({len(labels)})不匹配")
                    continue
                
                if total_length == 0:
                    continue
                
                # 滑动窗口处理
                # 窗口起点始终为0，每次增加总长度的10%
                step_size = max(1, int(total_length * 0.1))  # 至少增加1
                
                current_end = step_size
                while current_end <= total_length:
                    # 获取当前窗口的单词和标签
                    window_words = words[0:current_end]
                    window_labels = labels[0:current_end]
                    
                    # 计算得分：label值总和 / 长度
                    score = sum(window_labels) / len(window_labels)
                    
                    # 如果得分在0和1之间（不包括0和1），返回0.5
                    if 0 < score < 1:
                        score = 0.5
                    
                    # 组合窗口文本
                    window_text = " ".join(window_words)
                    
                    # 创建新的数据结构
                    result = {
                        "text": window_text,
                        "score": score
                    }
                    all_results.append(result)
                    
                    # 增加窗口大小
                    current_end += step_size
                
                # 确保最后一个窗口包含所有数据（如果还没包含的话）
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
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"已保存到 {output_file}")
    
    # 按4:1划分训练集和验证集
    train_data, val_data = train_test_split(all_results, test_size=0.2, random_state=42)
    
    # 保存训练集
    train_file = "data/train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存验证集
    val_file = "data/val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"训练集: {len(train_data)} 条数据，保存到 {train_file}")
    print(f"验证集: {len(val_data)} 条数据，保存到 {val_file}")
    
    # 显示一些统计信息
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
