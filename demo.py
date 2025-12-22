import json
import time
from pathlib import Path
from detection.samples import gen_samples


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
        try:
            texts, labels, out_ids, metas = gen_samples()
            save_records(output, texts, labels, out_ids, metas)
        except Exception:
            print("Error in gen_samples")
            time.sleep(2)
            continue
        time.sleep(1)


if __name__ == "__main__":
    main()