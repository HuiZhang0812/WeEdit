import os
import json
import time
import re
import logging
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# Configuration
# ============================================================
write_lock = Lock()


# ============================================================
# Implement your own Gmini-3-pro-Image API call here.
#
# This function should send a text prompt along with multiple
# images to a vision-language model (Gmini-3-pro-Image)
# and return the model's text response as a string.
# Returns:
#     str | None: The model's response text, or None on failure.
# ============================================================
def call_gemini(prompt: str, image_paths: list[str]) -> str | None:
    raise NotImplementedError(
        "Please implement `call_gemini()` with your own VLM API. "
        "It should accept a text prompt and a list of image paths, "
        "and return the model's response as a string."
    )


# ============================================================
# Prompt Construction
# ============================================================
TYPE_CRITERIA = {
    "add": """
- Check if the specified text has been added exactly as instructed
- Verify the text is placed at the correct position
- Ensure the added text content matches exactly what was requested
- Check if font style and size are appropriate for the context""",
    "delete": """
- Verify that the specified text has been completely removed
- Check that no traces or artifacts of the deleted text remain
- Ensure the area where text was deleted blends naturally with surroundings
- Confirm no unintended text was removed""",
    "replace": """
- Verify the original text has been completely replaced
- Check if the new text matches exactly what was specified
- Ensure the replacement text is in the same position as the original
- Verify font style consistency with surrounding text""",
    "translate": """
- Verify ALL text instances have been translated
- Check translation accuracy and appropriateness
- Ensure translated text maintains similar positioning
- Verify font is appropriate for the target language
- Check that no text was missed in translation""",
    "change_style": """
- Verify the text style has been changed as specified (font, color, bold, etc.)
- Check if the style change applies to all intended text
- Ensure the text content remains unchanged
- Verify the new style matches the description""",
    "combined": """
- Evaluate each individual edit instruction separately
- Check if ALL specified edits have been applied
- Verify no conflicts or inconsistencies between different edits
- Ensure all additions, deletions, and replacements are correctly executed""",
    "reasoning": """
- Verify the content has been updated according to the instruction
- Check factual accuracy of the new content
- Ensure all related elements are consistently updated
- Verify the update makes logical sense in context""",
    "rearrange": """
- Verify that the text has been moved to the specified new position
- Check that the text content remains unchanged after repositioning
- Ensure the new layout follows the specified arrangement instructions
- Verify proper alignment and spacing in the new position""",
}

EVAL_PROMPT_TEMPLATE = """You are an expert image editing evaluator. You will be shown two images:
- Image 1: The ORIGINAL source image (before editing)
- Image 2: The EDITED image (after model's editing attempt)

Your task is to evaluate the quality of the image editing based on the following instruction and criteria.

## Edit Instruction:
{edit_instruction}

## Operation Type: {instruction_type}

## Original Text (before editing):
{original_text}

## Target Text (expected after editing):
{target_text}

## Type-Specific Evaluation Criteria:
{type_criteria}

## Evaluation Dimensions:

### Dimension 1: Instruction Execution Accuracy (1-10 points)
- 10: Perfect execution, all aspects completed correctly
- 8-9: Minor issues, main instruction completed but small details missed
- 6-7: Partial completion, some parts done correctly
- 4-5: Significant issues, major parts not followed
- 0-3: Failed to execute or completely wrong result

### Dimension 2: Text Readability and Harmony (1-10 points)
- 10: Crystal clear text, perfect rendering, harmonious with image style
- 8-9: Clear text with minor imperfections, generally harmonious
- 6-7: Readable but noticeable issues (slight blur, font mismatch)
- 4-5: Difficult to read, significant blur or rendering issues
- 0-3: Illegible text or severely distorted rendering

### Dimension 3: Non-Edited Region Preservation (1-10 points)
- 10: Perfect preservation, no unintended changes
- 8-9: Minimal unintended changes, barely noticeable
- 6-7: Some unintended changes but overall structure preserved
- 4-5: Noticeable damage to non-edited regions
- 0-3: Significant damage or alteration to non-edited content

## Instructions:
1. Carefully analyze both images and identify all differences
2. For each dimension, provide detailed reasoning
3. Be objective and precise
4. Provide your final scores in the exact format below

## Response Format:

<scores>
instruction_accuracy: [score]
text_readability: [score]
preservation: [score]
</scores>

Where [score] is an integer from 1 to 10.

Now analyze the images and provide your evaluation:"""


def build_evaluation_prompt(item: dict) -> str:
    instruction_type = item.get("instruction_type", "unknown")
    return EVAL_PROMPT_TEMPLATE.format(
        edit_instruction=item.get("edit_instruction", ""),
        instruction_type=instruction_type,
        original_text=json.dumps(item.get("original_text", []), ensure_ascii=False),
        target_text=json.dumps(item.get("target_text", []), ensure_ascii=False),
        type_criteria=TYPE_CRITERIA.get(instruction_type, ""),
    )


# ============================================================
# Score Parsing
# ============================================================
def parse_scores(response_text: str) -> dict:
    scores = {"instruction_accuracy": None, "text_readability": None, "preservation": None}
    if not response_text:
        return scores
    match = re.search(r"<scores>(.*?)</scores>", response_text, re.DOTALL)
    if not match:
        return scores
    block = match.group(1)
    for key in scores:
        m = re.search(rf"{key}:\s*(\d+)", block)
        if m:
            scores[key] = int(m.group(1))
    return scores


# ============================================================
# File Utilities
# ============================================================
def resolve_generated_image(item: dict, generated_imgs_dir: Path) -> Path:
    img_id = item["img_info"]["img_id"]
    inst_type = item["instruction_type"]
    base = f"{img_id}_{inst_type}"
    for ext in (".png", ".jpg"):
        p = generated_imgs_dir / (base + ext)
        if p.exists():
            return p
    return generated_imgs_dir / (base + ".png")


def load_json(path: Path, default=None):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default if default is not None else []


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_failed_result(result: dict) -> bool:
    if not result:
        return True
    scores = result.get("scores", {})
    if all(scores.get(k) is None for k in ("instruction_accuracy", "text_readability", "preservation")):
        return True
    raw = result.get("raw_response", "")
    if "check freq error" in raw or "blocked by frequency check" in raw:
        return True
    return False


def result_key(r: dict) -> str:
    return f"{r.get('img_id', '')}_{r.get('instruction_type', '')}"


# ============================================================
# Single-Item Evaluation
# ============================================================
def evaluate_single_item(item: dict, index: int, generated_imgs_dir: Path) -> dict | None:
    try:
        source_path = Path("benchmark") / item["source_img_path"]
        generated_path = resolve_generated_image(item, generated_imgs_dir)

        if not source_path.exists():
            logging.error(f"[{index}] Source image not found: {source_path}")
            return None
        if not generated_path.exists():
            logging.error(f"[{index}] Generated image not found: {generated_path}")
            return None

        prompt = build_evaluation_prompt(item)
        response_text = call_gemini(prompt, [str(source_path), str(generated_path)])

        if not response_text:
            logging.error(f"[{index}] VLM call returned None")
            return None

        scores = parse_scores(response_text)
        result = {
            "index": index,
            "img_id": item["img_info"]["img_id"],
            "instruction_type": item["instruction_type"],
            "edit_instruction": item["edit_instruction"],
            "source_language": item.get("source_language", ""),
            "target_language": item.get("target_language", ""),
            "scores": scores,
            "raw_response": response_text,
        }
        logging.info(f"[{index}] Done - {result['img_id']} | type={result['instruction_type']} | scores={scores}")
        return result

    except Exception as e:
        logging.error(f"[{index}] Error: {e}")
        traceback.print_exc()
        return None


# ============================================================
# Statistics
# ============================================================
DIMS = ("instruction_accuracy", "text_readability", "preservation")


def compute_statistics(results: list[dict]) -> dict:
    valid = [r for r in results if r and r.get("scores")]

    dim_scores = {d: [] for d in DIMS}
    type_scores = {}

    for r in valid:
        s = r["scores"]
        t = r["instruction_type"]
        if t not in type_scores:
            type_scores[t] = {d: [] for d in DIMS}
        for d in DIMS:
            if s.get(d) is not None:
                dim_scores[d].append(s[d])
                type_scores[t][d].append(s[d])

    avg = lambda lst: round(sum(lst) / len(lst), 3) if lst else None

    stats = {
        "total_evaluated": len(valid),
        "dimension_averages": {d: avg(v) for d, v in dim_scores.items()},
        "type_averages": {},
        "overall_average": avg([s for v in dim_scores.values() for s in v]),
    }
    for t, td in type_scores.items():
        stats["type_averages"][t] = {
            "count": len(td[DIMS[0]]),
            **{d: avg(td[d]) for d in DIMS},
            "overall": avg([s for v in td.values() for s in v]),
        }
    return stats


# ============================================================
# Model Evaluation Loop
# ============================================================
def evaluate_model(results_dir: str, benchmark_file: str, max_workers: int = 8):
    results_dir = Path(results_dir)
    logging.info(f"\n{'='*60}\nEvaluating results in: {results_dir}\n{'='*60}")

    generated_imgs_dir = results_dir / "generated_imgs"
    results_file = results_dir / "evaluation_results.json"
    stats_file = results_dir / "evaluation_stats.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmark_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    logging.info(f"Loaded {len(items)} items")

    existing = load_json(results_file, [])
    done_keys = {result_key(r) for r in existing if r and not is_failed_result(r)}

    to_eval = [
        (i, item) for i, item in enumerate(items)
        if f"{item['img_info']['img_id']}_{item.get('instruction_type', 'unknown')}" not in done_keys
    ]
    logging.info(f"Already evaluated: {len(items) - len(to_eval)} | Remaining: {len(to_eval)}")

    if not to_eval:
        logging.info("All items already evaluated.")
        save_json(compute_statistics(existing), stats_file)
        return

    results = existing.copy()
    results_map = {result_key(r): i for i, r in enumerate(results) if r}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(evaluate_single_item, item, idx, generated_imgs_dir): (idx, item) for idx, item in to_eval}
        with tqdm(total=len(to_eval), desc=f"Evaluating") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    rk = result_key(res)
                    with write_lock:
                        if rk in results_map:
                            results[results_map[rk]] = res
                        else:
                            results_map[rk] = len(results)
                            results.append(res)
                        save_json(results, results_file)
                pbar.update(1)
                ok = sum(1 for r in results if r and not is_failed_result(r))
                pbar.set_postfix(success=ok, fail=len(results) - ok)

    save_json(results, results_file)
    stats = compute_statistics(results)
    save_json(stats, stats_file)

    logging.info(f"\n{'='*50}\nEvaluation Summary: {results_dir}\n{'='*50}")
    logging.info(f"Total evaluated: {stats['total_evaluated']}")
    for d, v in stats["dimension_averages"].items():
        logging.info(f"  {d}: {v}")
    logging.info(f"  Overall: {stats['overall_average']}")
    for t, ts in stats["type_averages"].items():
        logging.info(f"  {t} (n={ts['count']}): acc={ts['instruction_accuracy']} "
                     f"read={ts['text_readability']} pres={ts['preservation']} avg={ts['overall']}")
    logging.info(f"Results saved to: {results_file}")
    logging.info(f"Stats saved to:   {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate image editing results on bilingual benchmark")
    parser.add_argument("--results_dir", "-r", type=str, required=True,
                        help="Path to the results directory (should contain generated_imgs/ subfolder)")
    parser.add_argument("--benchmark_file", "-b", type=str, default="benchmark/Bilingual_benchmark.jsonl",
                        help="Path to the benchmark JSONL file (default: benchmark/Bilingual_benchmark.jsonl)")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    try:
        evaluate_model(args.results_dir, args.benchmark_file, args.workers)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

