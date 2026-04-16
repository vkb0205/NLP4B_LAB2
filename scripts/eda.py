import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "he", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "that", "the", "their", "them", "there", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "with", "you", "your", "im", "ive", "dont", "cant",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA on train/test intent datasets")
    parser.add_argument("--data-dir", type=str, default="sample_data", help="Directory containing train.csv and test.csv")
    parser.add_argument("--output-dir", type=str, default="reports/eda", help="Directory to save EDA outputs")
    parser.add_argument("--top-n", type=int, default=25, help="Top N words to include in word frequency tables")
    return parser.parse_args()


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["text"] = result["text"].astype(str)
    result["char_len"] = result["text"].str.len()
    result["word_len"] = result["text"].str.split().str.len()
    return result


def basic_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_text": int(df["text"].isna().sum()),
        "duplicate_text_rows": int(df["text"].duplicated().sum()),
        "unique_labels": int(df["label_id"].nunique()),
    }


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label_id"].value_counts().sort_index().rename("count")
    shares = (counts / counts.sum()).rename("share")
    return pd.concat([counts, shares], axis=1).reset_index(names="label_id")


def top_words(text_series: pd.Series, top_n: int) -> pd.DataFrame:
    token_counter = Counter()
    pattern = re.compile(r"[a-z0-9']+")

    for text in text_series.astype(str):
        tokens = pattern.findall(text.lower())
        filtered = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
        token_counter.update(filtered)

    top = token_counter.most_common(top_n)
    return pd.DataFrame(top, columns=["word", "count"])


def write_markdown_report(
    output_path: Path,
    train_summary: dict,
    test_summary: dict,
    train_len_stats: pd.DataFrame,
    test_len_stats: pd.DataFrame,
    drift: pd.DataFrame,
):
    report_file = output_path / "report.md"

    def table_to_markdown(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    with report_file.open("w", encoding="utf-8") as f:
        f.write("# EDA Report\n\n")

        f.write("## Dataset Overview\n\n")
        overview = pd.DataFrame([train_summary, test_summary], index=["train", "test"]).reset_index(names="split")
        f.write(table_to_markdown(overview))
        f.write("\n\n")

        f.write("## Text Length Statistics (Train)\n\n")
        f.write(table_to_markdown(train_len_stats.reset_index(names="metric")))
        f.write("\n\n")

        f.write("## Text Length Statistics (Test)\n\n")
        f.write(table_to_markdown(test_len_stats.reset_index(names="metric")))
        f.write("\n\n")

        f.write("## Label Distribution Shift (Train vs Test)\n\n")
        f.write(table_to_markdown(drift))
        f.write("\n")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected train.csv and test.csv in the data directory.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_columns = {"text", "label_id"}
    missing_train = required_columns - set(train_df.columns)
    missing_test = required_columns - set(test_df.columns)
    if missing_train:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"test.csv is missing required columns: {sorted(missing_test)}")

    train_df = add_text_features(train_df)
    test_df = add_text_features(test_df)

    train_summary = basic_summary(train_df)
    test_summary = basic_summary(test_df)

    train_len_stats = train_df[["char_len", "word_len"]].describe().T.round(2)
    test_len_stats = test_df[["char_len", "word_len"]].describe().T.round(2)

    train_labels = label_distribution(train_df).rename(columns={"count": "train_count", "share": "train_share"})
    test_labels = label_distribution(test_df).rename(columns={"count": "test_count", "share": "test_share"})

    drift = (
        train_labels.merge(test_labels, on="label_id", how="outer")
        .fillna(0)
        .assign(share_gap=lambda df: (df["train_share"] - df["test_share"]).abs())
        .sort_values("share_gap", ascending=False)
    )

    train_words = top_words(train_df["text"], args.top_n)
    test_words = top_words(test_df["text"], args.top_n)

    train_labels.to_csv(output_path / "train_label_distribution.csv", index=False)
    test_labels.to_csv(output_path / "test_label_distribution.csv", index=False)
    drift.to_csv(output_path / "label_share_gap.csv", index=False)
    train_len_stats.to_csv(output_path / "train_length_stats.csv")
    test_len_stats.to_csv(output_path / "test_length_stats.csv")
    train_words.to_csv(output_path / "train_top_words.csv", index=False)
    test_words.to_csv(output_path / "test_top_words.csv", index=False)

    write_markdown_report(
        output_path=output_path,
        train_summary=train_summary,
        test_summary=test_summary,
        train_len_stats=train_len_stats,
        test_len_stats=test_len_stats,
        drift=drift,
    )

    print(f"EDA complete. Outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
