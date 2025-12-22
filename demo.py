import json
import time
from pathlib import Path
from detection.samples import gen_samples

from loguru import logger
import sys
import time

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


def save_records(path, texts, labels, out_ids, metas): 
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out_set = set(out_ids.tolist()) if hasattr(out_ids, "tolist") else set(out_ids)
    with p.open("a", encoding="utf-8") as f:
        for i, (t, lbls, meta) in enumerate(zip(texts, labels, metas)):
            rec = {"text": t, "labels": [int(x) for x in lbls], "out_of_domain": bool(i in out_set), "data_source": meta.get("data_source"), "model_name": meta.get("model_name"), "model_params": meta.get("model_params")}
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


def main():
    print("Start generating samples")
    output = Path("data/generated_dataset_all.jsonl")
    while True:
        start_time = time.time()
        try:
            texts, labels, out_ids, metas = gen_samples()
            save_records(output, texts, labels, out_ids, metas)
            logger.info(f"Final Consume Time Each Round: Generated {len(texts)} samples in {time.time() - start_time:.2f} seconds")
        except Exception:
            logger.error("Error in gen_samples")
            time.sleep(2)
            continue
        time.sleep(1)


if __name__ == "__main__":
    setup_logger()
    main()