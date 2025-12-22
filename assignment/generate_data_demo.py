import itertools
import json
import multiprocessing as mp
import sys
import time
import types
from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from loguru import logger

from detection.validator.data_generator import DataGenerator
from detection.validator.text_completion import OllamaModel


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
    logger.add(
        "logs/error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n{exception}",
        level="ERROR",
        rotation="100 MB",
        retention="3 months",
    )


def _set_single_model(generator: DataGenerator, model: OllamaModel) -> None:
    generator.models = [model]
    generator.model_names = [model.model_name]
    generator.n_models = 1
    generator.n_models_with_in_the_middle = 1 if model.in_the_middle_generation else 0
    generator.models_with_in_the_middle = [0] if model.in_the_middle_generation else []


def _make_param_cycle():
    temps = [0.2, 0.7, 1.1, 1.6]
    repeat_penalties = [0.8, 1.0, 1.2, 1.4, 1.6]
    top_ps = [0.6, 0.8, 0.95]
    top_ks = [20, 40, 80]

    grid = [
        {"temperature": t, "repeat_penalty": rp, "top_p": tp, "top_k": tk}
        for t, rp, tp, tk in itertools.product(temps, repeat_penalties, top_ps, top_ks)
    ]
    return itertools.cycle(grid)


def _patch_init_model(model: OllamaModel, params_cycle) -> None:
    def _init_model(self):
        p = next(params_cycle)
        self.model = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            timeout=200,
            num_thread=1,
            num_predict=self.num_predict,
            temperature=p["temperature"],
            repeat_penalty=p["repeat_penalty"],
            top_p=p["top_p"],
            top_k=p["top_k"],
        )
        self.params = dict(p)

    model.init_model = types.MethodType(_init_model, model)


def _rows_to_records(rows):
    recs = []
    for el in rows:
        text = el.text_auged or el.text
        lbls = el.auged_segmentation_labels or el.segmentation_labels
        recs.append(
            {
                "text": text,
                "labels": [int(x) for x in lbls],
                "out_of_domain": el.data_source == "common_crawl",
                "data_source": el.data_source,
                "model_name": el.model_name,
                "model_params": el.model_params,
            }
        )
    return recs


def _write_jsonl_lines(f, records):
    dumps = json.dumps
    lines = [dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records]
    f.writelines(lines)


def _models_config(base_url: str):
    return [
        ("qwen:32b-text-v1.5-q4_0", False),
        ("qwen2.5-coder:32b", True),
        ("gemma2:27b-text-q4_0", False),
        ("gemma2:27b", True),
        ("mistral-small:22b", True),
        ("internlm2:20b", False),
        ("internlm/internlm2.5:20b-chat", False),
        ("deepseek-v2:16b", False),
        ("deepseek-r1:14b", False),
        ("phi4:14b", False),
        ("aya-expanse:32b", False),
        ("yi:34b-chat", False),
    ]


def worker_human(output_path: str, base_url: str, device: str, batch_size: int):
    setup_logger()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    name, in_mid = _models_config(base_url)[0]
    generator = DataGenerator([OllamaModel(model_name=name, base_url=base_url, in_the_middle_generation=in_mid)], device=device)

    with output.open("a", encoding="utf-8", buffering=1024 * 1024) as f:
        while True:
            start = time.time()
            try:
                rows = generator.generate_human_data(batch_size)
                _write_jsonl_lines(f, _rows_to_records(rows))
                logger.info(f"human: +{len(rows)} ({len(rows) / max(time.time() - start, 1e-6):.2f} rec/s)")
            except Exception:
                logger.exception("human worker failed")
                time.sleep(2)


def worker_ai(output_path: str, base_url: str, device: str, reinit_every: int, switch_every: int):
    setup_logger()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    models = [OllamaModel(model_name=n, base_url=base_url, in_the_middle_generation=mid) for n, mid in _models_config(base_url)]
    for m in models:
        _patch_init_model(m, _make_param_cycle())

    generator = DataGenerator([models[0]], device=device)

    with output.open("a", encoding="utf-8", buffering=1024 * 1024) as f:
        idx = 0
        while True:
            model = models[idx % len(models)]
            _set_single_model(generator, model)

            produced = 0
            while produced < switch_every:
                n = min(reinit_every, switch_every - produced)
                start = time.time()
                try:
                    rows = generator.generate_ai_data(n)
                    _write_jsonl_lines(f, _rows_to_records(rows))
                    produced += len(rows)
                    logger.info(
                        f"ai: {model.model_name} +{len(rows)} ({len(rows) / max(time.time() - start, 1e-6):.2f} rec/s) total={produced}/{switch_every}"
                    )
                except Exception:
                    logger.exception(f"ai worker failed on model={model.model_name}")
                    time.sleep(2)

            idx += 1


def worker_ai_in_middle(output_path: str, base_url: str, device: str, reinit_every: int, switch_every: int):
    setup_logger()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    configs = [(n, mid) for n, mid in _models_config(base_url) if mid]
    models = [OllamaModel(model_name=n, base_url=base_url, in_the_middle_generation=True) for n, _ in configs]
    for m in models:
        _patch_init_model(m, _make_param_cycle())

    generator = DataGenerator([models[0]], device=device)

    with output.open("a", encoding="utf-8", buffering=1024 * 1024) as f:
        idx = 0
        while True:
            model = models[idx % len(models)]
            _set_single_model(generator, model)

            produced = 0
            while produced < switch_every:
                n = min(reinit_every, switch_every - produced)
                start = time.time()
                try:
                    rows = generator.generated_ai_in_the_middle(n)
                    _write_jsonl_lines(f, _rows_to_records(rows))
                    produced += len(rows)
                    logger.info(
                        f"middle: {model.model_name} +{len(rows)} ({len(rows) / max(time.time() - start, 1e-6):.2f} rec/s) total={produced}/{switch_every}"
                    )
                except Exception:
                    logger.exception(f"middle worker failed on model={model.model_name}")
                    time.sleep(2)

            idx += 1


def main():
    base_url = "http://127.0.0.1:11434"
    device = "cuda"

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    procs = [
        mp.Process(
            name="human",
            target=worker_human,
            args=(str(out_dir / "generated_human.jsonl"), base_url, device, 200),
            daemon=False,
        ),
        mp.Process(
            name="ai",
            target=worker_ai,
            args=(str(out_dir / "generated_ai.jsonl"), base_url, device, 100, 1000),
            daemon=False,
        ),
        mp.Process(
            name="middle",
            target=worker_ai_in_middle,
            args=(str(out_dir / "generated_ai_middle.jsonl"), base_url, device, 100, 1000),
            daemon=False,
        ),
    ]

    for p in procs:
        p.start()

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()