# mission02-ai-content-detect

Trained_model and datasets: https://huggingface.co/xuqiang0905/ai_text_detect/tree/main

# 1. preprocess dataset
python assignment/data_preprocess.py --input_filepath=data/llama3_text/generated_dataset.json
# 2. merge train.jsonl and val.jsonl to processed_dataset.jsonl
python assignment/merge_dataset.py data/command/ data/gemma_9b/ data/llama3_text data/qwen2_14b --out-dir=data/merged_dataset
# 3. train model
python assignment/train.py --input_dir=data/merged_dataset --output_model_dir=model/2_trained_merged
# 4. evaluate model
python assignment/score.py --input_filepath=data/gemma_9b/generated_dataset_gemma_9b.jsonl --model_path=model/2_trained_merged/ --start_pos=20

# Introduction
Base_model: roberta-base-ai-text-detection-v1
Datasets: merged dataset of command, gemma_9b, llama3_text, qwen2_14b
Results: F1: 0.970162025352621
Evaluates: reward and out_of_domain_reward score: 0.95-1.0. I evaluated the model on every test set several times.
tips:
1. Segment the text with a sliding window where the window’s starting position remains unchanged.
2. If segmented text's labels are all 0, then the label of the segmented text is 0.
   If segmented text's labels are all 1, then the label of the segmented text is 2.
   If segmented text's labels are not all the same, then the label of the segmented text is 1.
3. During inference, the text is also segmented and predicted in batches. If the last predicted labels are either 0 or 2, the result is returned directly. Otherwise, identify the segments where the label is 1 and perform a second, finer-grained segmentation and prediction around those points. If all first-round segments are labeled as 1, then perform the second-round fine-grained segmentation on the first slice.