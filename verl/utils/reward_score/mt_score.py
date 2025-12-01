import re
from typing import Optional, Tuple, List
import sacrebleu

def compute_bleu(lang_pair: str, reference: str, prediction: str) -> float:
    """Compute BLEU score for a given language pair using sacrebleu."""
    prediction = prediction if isinstance(prediction, str) else ""
    src_lang, tgt_lang = lang_pair.split("-")

    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"
    bleu = sacrebleu.corpus_bleu([prediction], [[reference]], tokenize=tokenize)
    bleu_score = bleu.score

    return bleu_score


def extract_solution(text: str) -> str:
    """Extract the final translated segment from model output.
    
    Removes <think>...</think> blocks if present.
    """
    processed = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return processed.strip()


def validate_response_structure(response: str) -> bool:
    """Validate that the response follows the required tag structure."""
    print("\n" + "=" * 60)
    print(" Structure Validation ".center(60, "="))

    tags = {"think_start": "<think>", "think_end": "</think>"}
    validation_passed = True
    positions = {}

    for name, tag in tags.items():
        count = response.count(tag)
        pos = response.find(tag)
        positions[name] = pos
        status = "OK" if count == 1 else "ERROR"
        print(f"  {tag:12} | count={count:2} | position={pos:4} | {status}")

        if count != 1:
            validation_passed = False

    if any(positions[k] == -1 for k in positions):
        print("  [Error] Missing required tags.")
        return False

    if not (positions["think_start"] < positions["think_end"]):
        print("  [Error] Invalid tag sequence: <think> must come before </think>")
        validation_passed = False
    else:
        print("  Tag sequence order: OK")

    print("=" * 60 + "\n")
    return validation_passed


def compute_score(
    reward_metric: str,
    metric_scores: Optional[float] | List[float],
    lang_pair: str,
    solution_str: str,
    prompt_str: str,
    ground_truth: str,
    translation_raw: str,
    mul_times: int = 2,
    scale_factor: float = 100.0,
) -> float:
    """Compute total reward score based on model output and ground truth."""
    print("\n" + "=" * 80)
    print(" Processing Training Sample ".center(80, "="))

    # Validate original output
    format_correct = validate_response_structure(solution_str)

    # Extract final answer
    processed = extract_solution(solution_str)
    answer_text = processed.strip()

    print("\n" + "-" * 60)
    print(" Sample Info ".center(60, "-"))
    if translation_raw is None:
        print("[INFO] Stage 1 sample (no raw translation available).")
    else:
        min_score = min(metric_scores) if metric_scores else None
        print(f"[Answer]        {answer_text}")
        print(f"[Last Output]   {translation_raw}")
        print(f"[Min Comet]     {min_score}")
        print(f"[Answer==Raw]   {answer_text == translation_raw}")
        print(f"[Length]        answer={len(answer_text)}, raw={len(translation_raw)}")

        if answer_text == translation_raw and min_score < 82:
            print("[INFO] Low-quality duplicate, giving 0.0 score.")
            return 0.0
    print("-" * 60 + "\n")

    # Show prompt + response
    print(" Prompt + Model Output ".center(80, "-"))
    print(prompt_str + solution_str)
    print("-" * 80 + "\n")

    if not (format_correct and answer_text):
        print("[Content Validation] Skipped due to format error or empty answer.")
        return 0.0

    # Compute BLEU score
    bleu_score = compute_bleu(lang_pair, ground_truth, answer_text)
    answer_score = 0.0

    def scale(val: float) -> float:
        return val / scale_factor

    # Reward computation
    if reward_metric == "BLEU":
        answer_score = scale(bleu_score)
    elif reward_metric == "Model":
        if metric_scores is None:
            raise ValueError("metric_scores is None; expected model score.")
        answer_score = scale(sum(metric_scores))
    elif reward_metric == "Merge":
        if metric_scores is None:
            raise ValueError("metric_scores is None; expected model score.")
        answer_score = scale(bleu_score + sum(metric_scores))
    else:
        raise ValueError(f"Invalid reward_metric type: {reward_metric}")

    # Print content validation
    print("\n" + "=" * 60)
    print(" Content Validation ".center(60, "="))
    print(f"Reference      : {ground_truth}")
    print(f"Hypothesis     : {answer_text}")
    print(f"BLEU           : {bleu_score:.2f}")
    if metric_scores is not None:
        print(f"Model Score(s) : {metric_scores}")
    print("=" * 60)

    # Total score
    total_score = pow(answer_score, mul_times)
    print("\n" + "-" * 60)
    print(" Reward Breakdown ".center(60, "-"))
    print(f"  Answer Score : {answer_score:.4f}")
    print(f"  Total Score  : {total_score:.4f}")
    print("-" * 60 + "\n")

    return total_score



def compute_score_val_bleu(
    solution_str: str,
    ground_truth: str,
    lang_pair: str,
) -> float:
    """Compute BLEU score for validation samples (without reward logic)."""

    answer_text = extract_solution(solution_str)

    bleu_score = compute_bleu(lang_pair, ground_truth, answer_text or "")
    return bleu_score

def compute_score_corpus_bleu(
    solution_str: list[str],
    ground_truth: list[str],
    lang_pair: str
) -> float:
    assert len(solution_str) == len(ground_truth), (
        "The number of translations should be equal to the number of references"
    )
    _, tgt_lang = lang_pair.split("-")
    # Choose tokenizer based on target language
    if tgt_lang == "zh":
        tokenizer = "zh"
    elif tgt_lang == "ja":
        tokenizer = "ja-mecab"
    elif tgt_lang == "ko":
        tokenizer = "ko-mecab"
    else:
        tokenizer = "13a"
    print(
        f"Preview of responses and references:\nResponse: {solution_str[0]}\nReference: {ground_truth[0]}"
    )
    solution_str = [extract_solution(x) for x in solution_str]
    if any(x.startswith("<think>") for x in solution_str):
        print("Warning: Some responses still contain <think> tags after extraction.")
    result = sacrebleu.corpus_bleu(
        solution_str,
        [ground_truth],
        tokenize=tokenizer,
        force=True,
    )
    return result.score