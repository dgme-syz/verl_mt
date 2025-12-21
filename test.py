import re
import textwrap
import sacrebleu
import torch
from vllm import LLM, SamplingParams
from comet import download_model, load_from_checkpoint


# =========================
# Prompt builders
# =========================

def qwen_chat_input(src_lang_name, tgt_lang_name, src_text):
    user_input = (
        f"Translate the following text into {tgt_lang_name} "
        f"without additional explanations:\n\n"
        f"{src_text}\n\n"
    )
    return [{"role": "user", "content": user_input}]


def build_postedit_prompt(src_text, pred_text, target_lang, LANG_DICT):
    return textwrap.dedent(
        f"""
        Given the source text:

        {src_text}

        Improve the following draft {LANG_DICT.get(target_lang, target_lang)} translation
        into a high-quality {LANG_DICT.get(target_lang, target_lang)} version,
        without explanations:

        {pred_text}
        """
    ).strip()


# =========================
# Output preprocess
# =========================

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError(f"{text} is not a str.")
    parts = re.split(r'</think\s*>', text, flags=re.IGNORECASE)
    text_after_think = parts[-1] if len(parts) > 1 else text

    match = re.search(
        r'<text\s*>(.*?)</text\s*>',
        text_after_think,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        extracted = match.group(1)
    else:
        extracted = text_after_think
        if extracted.strip().startswith("<think>"):
            extracted = "null"

    return extracted.strip()


# =========================
# Metrics
# =========================

class chrFpp:
    def __call__(self, responses, references):
        result = sacrebleu.corpus_chrf(
            responses,
            [references],
            word_order=2,
            beta=2,
        )
        return result.score


def load_comet_kiwi():
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
    return load_from_checkpoint(model_path)


def comet_score(model, src, hyp, ref):
    data = [{"src": src, "mt": hyp, "ref": ref}]
    with torch.no_grad():
        return model.predict(data, batch_size=1, gpus=1)["scores"][0]


# =========================
# vLLM helpers
# =========================

def vllm_chat(llm, prompts, n, temperature=1.0):
    params = SamplingParams(
        temperature=temperature,
        n=n,
        max_tokens=3072,
    )
    outputs = llm.chat(prompts, params, use_tqdm=False)
    return [
        [preprocess(o.text) for o in out.outputs]
        for out in outputs
    ]


# =========================
# Main
# =========================

def main():
    # -------- config --------
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    N = 4   # translation rollouts
    M = 3   # post-edit rollouts

    src_text = "The Tierra del Sol Gallery is located at 7414 Santa Monica Blvd. For information, visit tierradelsolgallery.org."
    reference = "Tierra del Sol Gallery sijaitsee osoitteessa 7414 Santa Monica Blvd. Tietoa löytyy vierailemalla osoitteessa tierradelsolgallery.org."

    src_lang_name = "English"
    tgt_lang_name = "Finnish"
    target_lang = "fi"

    LANG_DICT = {"fi": "Finnish"}

    # -------- models --------
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        enable_thinking=True,
        trust_remote_code=True,
    )

    comet = load_comet_kiwi()
    chrf = chrFpp()

    # -------- translation rollout --------
    trans_prompt = qwen_chat_input(
        src_lang_name,
        tgt_lang_name,
        src_text,
    )

    translations = vllm_chat(
        llm,
        prompts=[trans_prompt],
        n=N,
        temperature=1.0,
    )[0]

    trajectories = []

    # -------- post-edit rollout --------
    for trans in translations:
        trans_chrf = chrf([trans], [reference])
        trans_comet = comet_score(comet, src_text, trans, reference)

        pe_prompt = build_postedit_prompt(
            src_text,
            trans,
            target_lang,
            LANG_DICT,
        )

        pe_outputs = vllm_chat(
            llm,
            prompts=[[{"role": "user", "content": pe_prompt}]],
            n=M,
            temperature=1.0,
        )[0]

        post_edits = []
        for pe in pe_outputs:
            post_edits.append({
                "text": pe,
                "chrfpp": chrf([pe], [reference]),
                "comet": comet_score(comet, src_text, pe, reference),
            })

        trajectories.append({
            "translation": {
                "text": trans,
                "chrfpp": trans_chrf,
                "comet": trans_comet,
            },
            "post_edits": post_edits,
        })

    # -------- pretty print --------
    print("\n" + "=" * 80)
    print("SOURCE:")
    print(src_text)
    print("REFERENCE:")
    print(reference)
    print("=" * 80)

    for i, traj in enumerate(trajectories):
        t = traj["translation"]
        print(f"\n[Translation #{i}]")
        print(f"chrF++={t['chrfpp']:.2f} | COMET={t['comet']:.4f}")
        print(t["text"])

        for j, pe in enumerate(traj["post_edits"]):
            print(f"\n  [Post-edit #{i}-{j}]")
            print(f"  chrF++={pe['chrfpp']:.2f} | COMET={pe['comet']:.4f}")
            print(f"  {pe['text']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
