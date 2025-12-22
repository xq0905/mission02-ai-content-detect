import json
import time
from pathlib import Path
from detection.validator.data_generator import DataGenerator
from detection.validator.text_completion import OllamaModel
from detection.validator.segmentation_processer import SegmentationProcesser
from detection.attacks.data_augmentation import DataAugmentator

from loguru import logger
import sys
import time
import os

import argparse

parser = argparse.ArgumentParser(description="Generate samples for AI content detection")
parser.add_argument("--type", type=str, required=True, help="human/ai/middle")

args = parser.parse_args()

ollama_url = "http://127.0.0.1:11434"

models = [
    # OllamaModel(model_name='llama2:13b', base_url=ollama_url, in_the_middle_generation=True),               #1
    # OllamaModel(model_name='llama3:text', base_url=ollama_url),                                           #0
    OllamaModel(model_name='llama3:70b', base_url=ollama_url, in_the_middle_generation=True),             #3
    OllamaModel(model_name='llama3.1:70b-text-q4_0', base_url=ollama_url, in_the_middle_generation=True), #3
    # OllamaModel(model_name='llama3.2', base_url=ollama_url, in_the_middle_generation=True),                 #1
    OllamaModel(model_name='llama3.3:70b', base_url=ollama_url),                                          #3                       


    # OllamaModel(model_name='qwen:32b-text-v1.5-q4_0', base_url=ollama_url),                               #2
    OllamaModel(model_name='qwen2:72b-text-q4_0', base_url=ollama_url),                                   #3
    # OllamaModel(model_name='qwen2.5:14b', base_url=ollama_url, in_the_middle_generation=True),            #0
    # OllamaModel(model_name='qwen2.5-coder:32b', base_url=ollama_url, in_the_middle_generation=True),      #2
    OllamaModel(model_name='qwen2.5:72b', base_url=ollama_url, in_the_middle_generation=True),            #3

    # OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),              #0
    # OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),              #0
    # OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),    #3
    # OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),    #3
    OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),    #3


    # OllamaModel(model_name='gemma2:9b-instruct-q4_0', base_url=ollama_url, in_the_middle_generation=True),#0
    # OllamaModel(model_name='gemma2:27b-text-q4_0', base_url=ollama_url),                                  #2
    # OllamaModel(model_name='gemma2:27b', base_url=ollama_url, in_the_middle_generation=True),             #2

    # OllamaModel(model_name='mistral:text', base_url=ollama_url),                                          #1
    # OllamaModel(model_name='mistral-nemo:12b', base_url=ollama_url, in_the_middle_generation=True),       #1
    # OllamaModel(model_name='mistral-small:22b', base_url=ollama_url, in_the_middle_generation=True),      #2
    OllamaModel(model_name='mistral-large:123b', base_url=ollama_url, in_the_middle_generation=True),     #3

    # OllamaModel(model_name='internlm2:7b', base_url=ollama_url),                                          #1
    # OllamaModel(model_name='internlm2:20b', base_url=ollama_url),                                         #2
    # OllamaModel(model_name='internlm/internlm2.5:20b-chat', base_url=ollama_url),                         #2

    # OllamaModel(model_name='deepseek-v2:16b', base_url=ollama_url),                                        #2
    # OllamaModel(model_name='deepseek-r1:14b', base_url=ollama_url),                                        #2
    # OllamaModel(model_name='phi4:14b', base_url=ollama_url),                                               #2
    # OllamaModel(model_name='aya-expanse:32b', base_url=ollama_url),                                        #2
    # OllamaModel(model_name='yi:34b-chat', base_url=ollama_url),                                            #2
    OllamaModel(model_name='athene-v2:72b', base_url=ollama_url),                                          #3
]
model_names = [model.model_name for model in models]
models_with_in_the_middle = [i for i in range(len(models)) if models[i].in_the_middle_generation]
models_names_with_in_the_middle = [model_names[i] for i in models_with_in_the_middle]

generator = DataGenerator(models, device="cuda")
augmentator = DataAugmentator(device="cuda")
segmentation_processer = SegmentationProcesser()


summary_prompts = [
    "Summarize the text in your own words, highlighting the key points. Do not generate anything else.",
    "Provide a concise summary of the text, focusing on its main argument. Refrain from generating anything else.",
    "In a few sentences, capture the core ideas of the text. Ensure you do not produce anything else.",
    "Write a short overview of the text, emphasizing the primary takeaways. Do not include anything else beyond the summary.",
    "Condense the text into a brief summary, touching on the essential details. Do not provide anything else in your response.",
    "Explain the text’s main points in a summarized format. Nothing else should be generated.",
    "Give me a succinct summary of the text’s content. Do not produce additional information.",
    "What is the most important information to include in a summary of this text? Only produce the summary, nothing else.",
    "Craft a concise review of the text, highlighting the central message. No other content should be added.",
    "Generate a quick summary that identifies the text’s key themes. Provide only the summary, with nothing else included.",
    "Offer a short synopsis of the text, noting the critical arguments. Please do not add anything else.",
    "Provide an executive summary of the text’s main findings. Avoid including extra information.",
    "Distill the text into a paragraph covering the core ideas. Refrain from adding any additional content.",
    "Summarize the text with an emphasis on its conclusion and supporting points. Do not provide anything beyond the summary.",
    "In just a few sentences, outline the text’s primary purpose and insights. Do not generate anything else."
]

generation_prompts = [
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will be given the start and finish of a text plus a summary of its middle. Your job is to compose only the middle portion, making sure it aligns with both the beginning and the end. Do not provide a summary; preserve any existing warnings by rephrasing them, and write nothing else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You receive the opening and closing paragraphs of a text, as well as a synopsis of the central section. Your task is to generate the text for the middle part alone, ensuring coherence with the given beginning and end. Keep any cautions or alerts by rewording them, and do not include any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You are provided with a text’s first and final segments along with a brief outline of what occurs in the middle. Your job is to fill in only the middle content. The final text should flow naturally, so do not insert a summary. Retain all warnings by rephrasing, and write nothing else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the initial and concluding parts of a text, plus a summary that describes the middle portion. Construct only the middle section so that it fits seamlessly from start to end. Rephrase and keep any warnings, and do not add a summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will see the beginning and ending of a text and a concise description of its midpoint. Your role is to write only the middle paragraphs, ensuring coherence with the provided segments. Maintain any disclaimers by rephrasing them, and avoid including any summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You get the start and finish of a text, as well as a summary of what happens in between. Create only the central portion, ensuring logical flow without adding a recap. Retain all cautions by rephrasing them as needed, and do not write anything else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. The opening and concluding lines of a text, plus a synopsis of the middle, will be given to you. Your aim is to produce only the middle content. Preserve existing warnings in a rephrased form, and refrain from including any summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the introduction and conclusion of a text, together with an overview of the midsection. Craft only the missing middle text so that the entire piece remains coherent. Keep any alerts or disclaimers by rewording them, and omit any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. Given the first and last parts of a text, plus a short summary of the middle, your job is to write only the central portion. Maintain coherence with the given sections, preserve any warnings by rephrasing them, and do not summarize anything. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will be shown the beginning and ending of a text along with a high-level summary of its midpoint. Only generate the middle content to ensure a continuous flow. Any existing notices or cautions must be included but reworded, and avoid all summarization. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You receive the text’s start and end segments, plus a brief overview of the middle. Construct only the middle text, ensuring it aligns with the summary and merges naturally with the given parts. Rephrase and keep warnings intact, without including a summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You are given the initial and final parts of a text, along with a concise account of what takes place in the middle. Write only the middle paragraphs to form a cohesive piece. Adjust any existing warnings but keep them, and exclude all summaries. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. The beginning and ending sections of a text, along with an abstract of the middle, are provided. Your task: generate only the missing central portion so the entire text reads coherently. Reword any cautions, and do not include any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the introduction and final segment of a text, plus a summary of the events in between. Craft the middle portion only, preserving flow. Keep warnings by restating them in your own words, and do not add any form of summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will see the first and last paragraphs of a text plus a synopsis of the middle. Provide only the middle text, ensuring it fits seamlessly. Retain any disclaimers by rephrasing them, and avoid providing any additional summarization. Do not generate anything else (Only middle part - you're output will be concatenated with begin and end)."
]

def regenerated_in_the_middle(model: OllamaModel, text, summary_prompt, generation_prompt):
    sentences = get_sentences(text)
    lens = [len(x) for x in sentences]
    first_part = len(sentences) // 3
    second_part = 2 * len(sentences) // 3
    third_part = len(sentences)

    first_size = sum(lens[:first_part])
    second_size = sum(lens[first_part:second_part])
    third_size = sum(lens[second_part:])
    for i in range(10):
        if first_size - lens[first_part - 1] > second_size + lens[first_part - 1]:
            first_part -= 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_size - lens[second_part - 1] > third_size + lens[second_part - 1]:
            second_part -= 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        elif first_size + lens[first_part] < second_size - lens[first_part]:
            first_part += 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_size + lens[second_part] < third_size - lens[second_part]:
            second_part += 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        else:
            break

    begin = ''.join(sentences[:first_part])
    middle = ''.join(sentences[first_part:second_part])
    end = ''.join(sentences[second_part:])

    middle_stripped = middle.rstrip()
    diff = len(middle) - len(middle_stripped)
    end = middle[-diff:] + end
    middle = middle_stripped

    assert model.in_the_middle_generation
    summary = model.classic_invoke([
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": middle}
    ])
    middle_size = len(middle.split())
    generated_middle = model.classic_invoke([
        {"role": "system", "content": generation_prompt + f" The middle should be about {middle_size} words long"},
        {"role": "user", "content": f"begin: {begin}\nend: {end}\nsummary: {summary}"}
    ])
    labels = [0] * len(begin.split()) + [1] * len(generated_middle.strip().split()) + [0] * len(end.split())
    return begin + generated_middle.strip() + end, labels

# 配置日志
def setup_logger():
    # 移除默认配置
    logger.remove()
    
    # 控制台输出（彩色）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # 文件输出（结构化 JSON）
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.jsonl",
        format="{message}",
        serialize=True,  # 输出 JSON
        rotation="00:00",  # 每天午夜轮转
        retention="1 month",
        compression="zip",
        level="DEBUG"
    )
    
    # 错误日志单独文件
    logger.add(
        "logs/error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n{exception}",
        level="ERROR",
        rotation="100 MB",
        retention="3 months"
    )

def save_records(path, texts, labels, out_ids=None, metas=None): 
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # out_set = set(out_ids.tolist()) if hasattr(out_ids, "tolist") else set(out_ids)

    if metas is None and out_ids is None:
        with p.open("a", encoding="utf-8") as f:
            for i, (t, lbls) in enumerate(zip(texts, labels)):
                print(f"lbls: {lbls}")
                rec = {"text": t, "labels": [int(x) for x in lbls]}
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")

def worker_human():
    logger.info(f"Start Generating Samples Of Human Data")
    record_filepath = "data/generated_dataset_human.jsonl"

    os.makedirs(Path(record_filepath).parent, exist_ok=True)

    texts_l = []
    labels_l = []
    min_text_length = 250
    while True:
        el = next(generator.human_dataset)

        text, cnt_first_human = el['text'], len(el['text'].split())
        el['segmentation_labels'] = cnt_first_human * [0]
        labels = el['segmentation_labels']

        text, labels = segmentation_processer.subsample_words(text, labels)
        if len(labels) == 0:
            continue

        text_auged, augs, labels_auged = augmentator(text, labels)

        if min_text_length <= len(text_auged):
            # el['text_auged'] = text_auged
            # el['augmentations'] = augs
            # el['auged_segmentation_labels'] = labels_auged

            texts_l.append(text_auged)
            labels_l.append(labels_auged)
            print(f"labels_auged: {labels_auged}")

        if len(texts_l) >= 100:
            save_records(record_filepath, texts_l, labels_l)
            texts_l = []
            labels_l = []

def worker_ai():
        logger.info(f"Start Generating Samples Of AI Data")
        record_filepath = "data/generated_dataset_ai.jsonl"

        i = 0
        model_cnt = 0
        init_cnt = 0
        texts_l = []
        labels_l = []
        min_text_length = 250

        model = models[i]
        model_name = model_names[i]
        logger.info(f"Start Generating Samples Of AI Data With Model {model_name}")
        while True:
            model_cnt += 1
            if model_cnt > 1000:
                model_cnt = 0
                i = (i + 1) % len(models)
                model = models[i]
                model_name = model_names[i]
                logger.info(f"Start Generating Samples Of AI Data With Model {model_name}")

            init_cnt += 1
            if init_cnt > 100:
                init_cnt = 0
                model.init_model()
                logger.info(f"Model {model_name} initialized")

            el = next(generator.prompt_dataset)
            el['completion'] = model(el['prompt'], text_completion_mode=True)
            el['model_name'] = model_name
            el['model_params'] = model.params

            text, cnt_first_human = segmentation_processer.merge_prompt_text(el['prompt'], el['completion'])
            labels = [0] * cnt_first_human + [1] * (len(text.split()) - cnt_first_human)
            el['text'] = text
            el['segmentation_labels'] = labels

            text, labels = segmentation_processer.subsample_words(text, labels)
            if len(labels) == 0:
                continue

            try:
                text_auged, augs, labels_auged = augmentator(text, labels)
                assert len(text_auged.split()) == len(labels_auged)
            except:
                logger.error("Got error during augmentations for text: {} \n and labels: {}".format(text, labels))
                logging.info(traceback.format_exc())
                continue

            if min_text_length <= len(text_auged):
                texts_l.append(text_auged)
                labels_l.append(labels_auged)

            if len(texts_l) >= 10:
                logger.info(f"Generated 10 samples of AI data with model {model_name}")
                save_records(record_filepath, texts_l, labels_l)
                texts_l = []
                labels_l = []

def generated_ai_in_the_middle():
        logger.info(f"Start Generating Samples Of Middle Data")
        record_filepath = "data/generated_dataset_middle.jsonl"

        i = 0
        model_cnt = 0
        init_cnt = 0
        texts_l = []
        labels_l = []
        min_text_length = 250

        model = models_with_in_the_middle[i]
        model_name = models_names_with_in_the_middle[i]
        logger.info(f"Start Generating Samples Of Middle Data With Model {model_name}")
        while True:
            model_cnt += 1
            if model_cnt > 1000:
                model_cnt = 0
                i = (i + 1) % len(models_with_in_the_middle)
                model = models_with_in_the_middle[i]
                model_name = models_names_with_in_the_middle[i]
                logger.info(f"Start Generating Samples Of Middle Data With Model {model_name}")

            init_cnt += 1
            if init_cnt > 100:
                init_cnt = 0
                model.init_model()
                logger.info(f"Model {model_name} initialized")
            try:
                el = None
                while el is None:
                    el = next(generator.prompt_dataset)
                    if len(nltk.sent_tokenize(el['prompt'])) < 3 or len(el['prompt'].split()) < min_text_length * 0.75:
                        el = None
                        continue
                summary_idx = np.random.randint(len(summary_prompts))
                generation_idx = np.random.randint(len(generation_prompts))
                text, labels = regenerated_in_the_middle(model, el['prompt'], summary_prompts[summary_idx], generation_prompts[generation_idx])
                el['text'] = text
                el['segmentation_labels'] = labels
                el['model_name'] = model_name
                el['model_params'] = model.params

                text, labels = segmentation_processer.subsample_words(text, labels)
                if len(labels) == 0:
                    continue

                try:
                    text_auged, augs, labels_auged = augmentator(text, labels)
                    assert len(text_auged.split()) == len(labels_auged)
                except:
                    logger.error("Got error during augmentations for text: {} \n and labels: {}".format(text, labels))
                    logging.info(traceback.format_exc())
                    continue

                if min_text_length <= len(text_auged):
                    texts_l.append(text_auged)
                    labels_l.append(labels_auged)
                
                if len(texts_l) >= 10:
                    logger.info(f"Generated 10 samples of AI data with model {model_name}")
                    save_records(record_filepath, texts_l, labels_l)
                    texts_l = []
                    labels_l = []

            except Exception as e:
                logger.error(f"Error during generation with {model_name} model: {e}")
                logging.info(traceback.format_exc())
                continue


def main():
    if args.type == "human":
        worker_human()
    elif args.type == "ai":
        worker_ai()
    elif args.type == "middle":
        worker_middle("data/generated_dataset_middle.jsonl", "http://localhost:11434", "cuda", 10)
    else:
        logger.error("Invalid type. Please choose from human, ai, or middle.")
        return


if __name__ == "__main__":
    setup_logger()
    main()