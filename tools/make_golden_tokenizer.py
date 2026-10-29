"""Generate the golden file for the Swift WordPieceTokenizer test.

Runs the Hugging Face tokenizer of the fine-tuned checkpoint over a fixed list of phrases and stores the
expected `input_ids` and `attention_mask`. The Swift test (`WordPieceTokenizerGoldenTests`) must reproduce
them exactly.

The app lowercases Russian input before tokenizing (`Language.lowercasesInput`), because the checkpoint's
tokenizer has `lowercase: False`. The golden file mirrors that: ids are computed for `text.lower()`.

Usage:
    python tools/make_golden_tokenizer.py --model_dir <checkpoint> \
        [--vocab Packages/MealRecognize/Sources/MealRecognize/Russian/Resources/vocab.txt] \
        [--output Packages/MealRecognize/Tests/MealRecognizeTests/Resources/golden_tokenizer.json]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOCAB = ROOT / "Packages/MealRecognize/Sources/MealRecognize/Russian/Resources/vocab.txt"
DEFAULT_OUTPUT = ROOT / "Packages/MealRecognize/Tests/MealRecognizeTests/Resources/golden_tokenizer.json"
SEQ_LEN = 128

PHRASES = [
    # typical meal phrases
    "вчера на ужин курица 150 грамм и гречка 80 грамм",
    "150 грамм курицы 80 грамм гречки в семь вечера",
    "на завтрак овсянка 200 грамм и банан",
    "сегодня в обед борщ 300 миллилитров",
    "съел два яйца и бутерброд с сыром",
    "позавчера в 8:30 кофе с молоком 250 мл",
    "на перекус яблоко и горсть орехов",
    "добавь творог 5% 180 г",
    "утром выпил стакан воды и съел тост",
    "в полдник йогурт 125 грамм",
    "на ужин паста с томатным соусом 350 грамм и салат",
    "сто пятьдесят грамм риса",
    # case, whitespace, empty
    "ВЧЕРА НА УЖИН Курица 150 ГРАММ",
    "Вчера На Ужин Гречка",
    "  много   пробелов \t и\nпереносы   строк  ",
    "",
    "   ",
    # digits and punctuation
    "150",
    "1,5 литра молока",
    "курица, 150 г.; гречка - 80 г!",
    "кофе-машина и какао-порошок",
    "«борщ» (300 мл) / хлеб",
    "время 12:45 или 12.45",
    "цена 99.9 руб. №5",
    "50% скидка + 2 яйца = завтрак",
    # letters
    "ёжик и ёлка, всё съел",
    "шоколад milka и чай lipton",
    "coffee with milk",
    "йогурт «Активиа» 4,5%",
    # unknown / rare
    "🍕 пицца 🍔",
    "авокадо тост с лососем и рукколой",
    "кинза, петрушка, укроп",
    "кефир 1%",
    "антидисэстаблишментарианство",
    "ооооочень вкусная еда",
    "тире — длинное и – среднее",
    # long, hits the 128-piece limit
    " ".join(["курица 150 грамм и гречка 80 грамм"] * 12),
    " ".join(f"слово{i}" for i in range(80)),
]


def check_vocab(tokenizer, vocab_path):
    repo_vocab = {line.rstrip("\n"): i for i, line in enumerate(vocab_path.read_text(encoding="utf-8").split("\n")) if line}
    hf_vocab = tokenizer.get_vocab()
    if repo_vocab != hf_vocab:
        only_repo = len(set(repo_vocab.items()) - set(hf_vocab.items()))
        only_hf = len(set(hf_vocab.items()) - set(repo_vocab.items()))
        sys.exit(f"vocab.txt in the repo differs from the checkpoint vocabulary "
                 f"({only_repo} entries only in repo, {only_hf} only in checkpoint)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Local checkpoint or Hugging Face hub id")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    check_vocab(tokenizer, args.vocab)

    cases = []
    for text in PHRASES:
        encoded = tokenizer(
            text.lower(), max_length=SEQ_LEN, padding="max_length", truncation=True,
        )
        cases.append({
            "text": text,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"max_length": SEQ_LEN, "cases": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
