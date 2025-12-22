import numpy as np

from detection.validator.data_generator import DataGenerator
from detection.validator.text_completion import OllamaModel
from detection.validator.segmentation_processer import SegmentationProcesser
from detection.attacks.data_augmentation import DataAugmentator


ollama_url = "http://127.0.0.1:11434"

models = [
    # OllamaModel(model_name='llama2:13b', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='llama3:text', base_url=ollama_url),
    # OllamaModel(model_name='llama3:70b', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='llama3.1:70b-text-q4_0', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='llama3.2', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='llama3.3:70b', base_url=ollama_url),


    OllamaModel(model_name='qwen:32b-text-v1.5-q4_0', base_url=ollama_url),
    # OllamaModel(model_name='qwen2:72b-text-q4_0', base_url=ollama_url),
    # OllamaModel(model_name='qwen2.5:14b', base_url=ollama_url, in_the_middle_generation=True),
    OllamaModel(model_name='qwen2.5-coder:32b', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='qwen2.5:72b', base_url=ollama_url, in_the_middle_generation=True),

    # OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),


    # OllamaModel(model_name='gemma2:9b-instruct-q4_0', base_url=ollama_url, in_the_middle_generation=True),
    OllamaModel(model_name='gemma2:27b-text-q4_0', base_url=ollama_url),
    OllamaModel(model_name='gemma2:27b', base_url=ollama_url, in_the_middle_generation=True),

    # OllamaModel(model_name='mistral:text', base_url=ollama_url),
    # OllamaModel(model_name='mistral-nemo:12b', base_url=ollama_url, in_the_middle_generation=True),
    OllamaModel(model_name='mistral-small:22b', base_url=ollama_url, in_the_middle_generation=True),
    # OllamaModel(model_name='mistral-large:123b', base_url=ollama_url, in_the_middle_generation=True),

    # OllamaModel(model_name='internlm2:7b', base_url=ollama_url),
    OllamaModel(model_name='internlm2:20b', base_url=ollama_url),
    OllamaModel(model_name='internlm/internlm2.5:20b-chat', base_url=ollama_url),

    OllamaModel(model_name='deepseek-v2:16b', base_url=ollama_url),
    OllamaModel(model_name='deepseek-r1:14b', base_url=ollama_url),
    OllamaModel(model_name='phi4:14b', base_url=ollama_url),
    OllamaModel(model_name='aya-expanse:32b', base_url=ollama_url),
    OllamaModel(model_name='yi:34b-chat', base_url=ollama_url),
    # OllamaModel(model_name='athene-v2:72b', base_url=ollama_url),
]

generator = DataGenerator(models, device="cuda")
augmentator = DataAugmentator(device="cuda")
segmentation_processer = SegmentationProcesser()


def gen_samples():
    data = generator.generate_data(n_human_samples=3, n_ai_samples=9)
    queries = [el for el in data]
    out_of_domain_ids = np.where([el.data_source == 'common_crawl' for el in queries])[0]
    auged_texts = []
    auged_labels = []
    metas = []
    for el in queries:
        text, lbls = segmentation_processer.subsample_words(el.text, el.segmentation_labels)
        new_text, augs, new_labels = augmentator(text, lbls)
        if len(new_text) >= 250:
            auged_texts.append(new_text)
            auged_labels.append(new_labels)
        else:
            auged_texts.append(el.text_auged)
            auged_labels.append(el.auged_segmentation_labels)
        metas.append({'data_source': el.data_source, 'model_name': el.model_name, 'model_params': el.model_params})
    return auged_texts, auged_labels, out_of_domain_ids, metas