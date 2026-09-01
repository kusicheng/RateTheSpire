# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .data_io import write_ndjson
from .trainers import train_model

def make_synthetic_records(n: int = 1000, seed: int = 7) -> list[dict[str, Any]]:
    """Create a toy dataset for smoke tests only."""
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        asc = rng.randint(0, 20)
        max_hp = rng.choice([70, 75, 80])
        hp = rng.randint(1, max_hp)
        gold = rng.randint(0, 450)
        path_len = rng.randint(5, 15)
        path = [rng.choice(["M", "?", "E", "R", "$", "T"]) for _ in range(path_len)]
        score = 0.0
        score += 0.9 * path.count("E")
        score += 0.7 * path.count("R")
        score += 0.2 * path.count("?")
        score += 0.4 if "$" in path and gold >= 180 else 0.0
        score -= 1.0 if hp / max_hp < 0.35 and path.count("E") >= 2 else 0.0
        score -= 0.05 * asc
        label = score >= 2.2
        records.append(
            {
                "character": rng.choice(["IRONCLAD", "SILENT", "DEFECT", "WATCHER"]),
                "ascension": asc,
                "act": rng.randint(1, 3),
                "floor": rng.randint(0, 45),
                "current_hp": hp,
                "max_hp": max_hp,
                "current_gold": gold,
                "deck": ["Strike"] * rng.randint(5, 25),
                "relics": ["Relic"] * rng.randint(0, 12),
                "potions": ["Potion"] * rng.randint(0, 3),
                "candidate_path": path,
                "victory": label,
            }
        )
    return records


def self_test(args: argparse.Namespace) -> dict[str, Any]:
    temp_path = Path(args.output_dir) / "synthetic_decisions.ndjson"
    records = make_synthetic_records(n=args.rows, seed=args.seed)
    write_ndjson(temp_path, records)
    train_args = argparse.Namespace(
        input=str(temp_path),
        model_out=str(Path(args.output_dir) / "synthetic_model.pkl"),
        report_out=str(Path(args.output_dir) / "synthetic_report.json"),
        max_rows=None,
        test_size=0.2,
        val_size=0.2,
        seed=args.seed,
        d_model=64,
        lr=0.001,
        batch_size=64,
        epochs=40,
        patience=8,
        aux_floor_weight=0.2,
        pos_weight="auto",
        device="auto",
        no_position=False,
        no_attention_pooling=False,
        split_strategy="stratified-group",
    )
    return train_model(train_args)
