# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .blend import blend_probability_arrays
from .data_io import parse_path, read_records
from .datasets import PathDecisionDataset, as_tabular_matrix
from .features import FeatureConfig
from .model_io import bundle_config, load_raw_bundle
from .neural import PathRaterNet
from .splits import row_label, split_records, split_report

def collect_probabilities(model: nn.Module, loader: DataLoader) -> tuple[list[float], list[int]]:
    model.eval()
    device = next(model.parameters()).device
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            numeric, rooms, cats, batch_labels = batch[:4]
            numeric = numeric.to(device)
            rooms = rooms.to(device)
            cats = cats.to(device)
            logits = model(numeric, rooms, cats)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch_labels.cpu().numpy().astype(int).tolist())
    return probs, labels


def best_threshold(model: nn.Module, loader: DataLoader) -> float:
    probs, labels = collect_probabilities(model, loader)
    return best_threshold_from_probs(probs, labels)


def best_threshold_from_probs(probs: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(probs, labels), key=lambda item: item[0], reverse=True)
    if not pairs:
        return 0.5
    positives = sum(1 for label in labels if int(label) == 1)
    true_positives = 0
    false_positives = 0
    best_score = -1.0
    best_threshold_value = 0.5
    idx = 0
    while idx < len(pairs):
        threshold_value = float(pairs[idx][0])
        while idx < len(pairs) and float(pairs[idx][0]) == threshold_value:
            if int(pairs[idx][1]) == 1:
                true_positives += 1
            else:
                false_positives += 1
            idx += 1
        false_negatives = positives - true_positives
        denominator = 2 * true_positives + false_positives + false_negatives
        score = (2 * true_positives / denominator) if denominator else 0.0
        if score > best_score:
            best_score = score
            best_threshold_value = threshold_value
    return best_threshold_value


def evaluate_model(model: nn.Module, loader: DataLoader, threshold: float = 0.5) -> dict[str, Any]:
    probs, labels = collect_probabilities(model, loader)
    return evaluate_probabilities(probs, labels, threshold=threshold)


def evaluate_predictions(labels: list[int], preds: list[int]) -> dict[str, Any]:
    report = classification_report(labels, preds, zero_division=0, output_dict=True)
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "positive_f1": float(report.get("1", {}).get("f1-score", 0.0)),
        "negative_f1": float(report.get("0", {}).get("f1-score", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "accuracy": float(report.get("accuracy", 0.0)),
        "classification_report": report,
        "n": len(labels),
    }


def evaluate_probabilities(probs: list[float], labels: list[int], threshold: float = 0.5) -> dict[str, Any]:
    preds = [1 if prob >= threshold else 0 for prob in probs]
    return evaluate_predictions(labels, preds)


def predict_dataset_probabilities(bundle: dict[str, Any], data: PathDecisionDataset) -> list[float]:
    if bundle.get("model_type") in {"hist_gradient_boosting", "extra_trees"}:
        matrix, _labels = as_tabular_matrix(data)
        return bundle["model"].predict_proba(matrix)[:, 1].tolist()
    if bundle.get("model_type") == "hist_gradient_boosting_ensemble":
        matrix, _labels = as_tabular_matrix(data)
        prob_arrays = [model.predict_proba(matrix)[:, 1] for model in bundle["models"]]
        return blend_probability_arrays(prob_arrays, bundle.get("blend_weights", [])).tolist()

    config = data.config
    model_args = bundle.get("model_args", {})
    model = PathRaterNet(
        numeric_dim=len(config.numeric_fields),
        cat_cardinalities=[len(config.categorical_values[field]) for field in CATEGORICAL_FIELDS],
        d_model=model_args.get("d_model", 64),
        use_position=model_args.get("use_position", False),
        use_attention_pooling=model_args.get("use_attention_pooling", False),
    )
    model.load_state_dict(bundle["model_state"], strict=False)
    return collect_probabilities(model, DataLoader(data, batch_size=1024))[0]


def numeric_bucket(value: float, cuts: list[tuple[float, str]], default: str) -> str:
    for limit, label in cuts:
        if value <= limit:
            return label
    return default


def slice_value(record: dict[str, Any], field_name: str) -> str:
    if field_name in {"character", "skill_bucket"}:
        return str(record.get(field_name, "UNKNOWN"))
    if field_name == "act":
        return f"act_{int(float(record.get('act', 0) or 0))}"
    if field_name == "ascension_bucket":
        asc = float(record.get("ascension", 0) or 0)
        return numeric_bucket(asc, [(0, "a0"), (9, "a1_9"), (14, "a10_14"), (19, "a15_19")], "a20")
    if field_name == "floor_band":
        floor = float(record.get("floor", 0) or 0)
        return numeric_bucket(floor, [(16, "floor_0_16"), (33, "floor_17_33")], "floor_34_plus")
    if field_name == "hp_bucket":
        hp = float(record.get("current_hp", 0) or 0)
        max_hp = float(record.get("max_hp", 0) or 0)
        hp_ratio = hp / max_hp if max_hp > 0 else 0.0
        return numeric_bucket(hp_ratio, [(0.35, "hp_low"), (0.7, "hp_mid")], "hp_high")
    if field_name == "gold_bucket":
        gold = float(record.get("current_gold", 0) or 0)
        return numeric_bucket(gold, [(99, "gold_low"), (249, "gold_mid")], "gold_high")
    if field_name == "path_len_bucket":
        path_len = len(parse_path(record.get("candidate_path", [])))
        return numeric_bucket(float(path_len), [(5, "path_1_5"), (10, "path_6_10")], "path_11_plus")
    return str(record.get(field_name, "UNKNOWN"))


def threshold_group_key(record: dict[str, Any], fields: list[str]) -> str:
    return "|".join(f"{field_name}={slice_value(record, field_name)}" for field_name in fields)


def global_threshold_settings(threshold: float) -> dict[str, Any]:
    return {"mode": "global", "threshold": float(threshold), "fallback_threshold": float(threshold), "rules": []}


def slice_threshold_settings(
    probs: list[float],
    rows: list[dict[str, Any]],
    labels: list[int],
    min_rows: int = 500,
) -> dict[str, Any]:
    fallback = best_threshold_from_probs(probs, labels)
    rules: list[dict[str, Any]] = []
    for fields in THRESHOLD_GROUP_SPECS:
        grouped: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            grouped.setdefault(threshold_group_key(row, fields), []).append(idx)
        thresholds: dict[str, float] = {}
        counts: dict[str, int] = {}
        for key, indices in sorted(grouped.items()):
            if len(indices) < min_rows:
                continue
            slice_labels = [labels[idx] for idx in indices]
            if len(set(slice_labels)) < 2:
                continue
            thresholds[key] = best_threshold_from_probs([probs[idx] for idx in indices], slice_labels)
            counts[key] = len(indices)
        if thresholds:
            rules.append({"fields": list(fields), "thresholds": thresholds, "counts": counts})
    return {
        "mode": "slice",
        "fallback_threshold": float(fallback),
        "min_rows": int(min_rows),
        "rules": rules,
    }


def make_threshold_settings(
    mode: str,
    probs: list[float],
    rows: list[dict[str, Any]],
    labels: list[int],
    min_rows: int = 500,
) -> dict[str, Any]:
    if mode == "slice":
        return slice_threshold_settings(probs, rows, labels, min_rows=min_rows)
    return global_threshold_settings(best_threshold_from_probs(probs, labels))


def threshold_for_row(row: dict[str, Any], settings: dict[str, Any]) -> float:
    if not settings or settings.get("mode") != "slice":
        return float(settings.get("threshold", settings.get("fallback_threshold", 0.5))) if settings else 0.5
    for rule in settings.get("rules", []):
        fields = list(rule.get("fields", []))
        key = threshold_group_key(row, fields)
        if key in rule.get("thresholds", {}):
            return float(rule["thresholds"][key])
    return float(settings.get("fallback_threshold", 0.5))


def predictions_from_threshold_settings(
    probs: list[float],
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[int]:
    return [1 if prob >= threshold_for_row(row, settings) else 0 for prob, row in zip(probs, rows)]


def evaluate_probabilities_with_threshold_settings(
    probs: list[float],
    rows: list[dict[str, Any]],
    labels: list[int],
    settings: dict[str, Any],
) -> dict[str, Any]:
    preds = predictions_from_threshold_settings(probs, rows, settings)
    return evaluate_predictions(labels, preds)


def analyze_prediction_slices(
    model_path: Path,
    input_path: Path,
    split_strategy: str = "stratified-group",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 7,
    min_rows: int = 200,
) -> dict[str, Any]:
    bundle = load_raw_bundle(model_path)
    config = bundle_config(bundle)
    records = read_records(input_path)
    train_rows, val_rows, test_rows = split_records(
        records,
        test_size,
        val_size,
        seed,
        strategy=split_strategy,
    )
    test_data = PathDecisionDataset(test_rows, config)
    probs = predict_dataset_probabilities(bundle, test_data)
    labels = [row_label(row) for row in test_rows]
    threshold = float(bundle.get("metrics", {}).get("threshold", 0.5))
    threshold_settings = bundle.get("threshold_settings") or global_threshold_settings(threshold)
    overall = evaluate_probabilities_with_threshold_settings(probs, test_rows, labels, threshold_settings)
    slice_fields = [
        "character",
        "act",
        "ascension_bucket",
        "floor_band",
        "hp_bucket",
        "gold_bucket",
        "path_len_bucket",
        "skill_bucket",
    ]
    slices: dict[str, dict[str, Any]] = {}
    for field_name in slice_fields:
        grouped: dict[str, list[int]] = {}
        for idx, row in enumerate(test_rows):
            grouped.setdefault(slice_value(row, field_name), []).append(idx)
        field_result: dict[str, Any] = {}
        for value, indices in sorted(grouped.items()):
            if len(indices) < min_rows:
                continue
            slice_probs = [probs[idx] for idx in indices]
            slice_labels = [labels[idx] for idx in indices]
            slice_rows = [test_rows[idx] for idx in indices]
            metrics = evaluate_probabilities_with_threshold_settings(
                slice_probs,
                slice_rows,
                slice_labels,
                threshold_settings,
            )
            metrics["positive_rate"] = float(sum(slice_labels) / len(slice_labels))
            field_result[value] = metrics
        slices[field_name] = field_result
    run_ids = {str(row.get("run_id", idx)) for idx, row in enumerate(records)}
    return {
        "model": str(model_path),
        "input": str(input_path),
        "split_strategy": split_strategy,
        "rows": len(records),
        "runs": len(run_ids),
        "test_rows": len(test_rows),
        "threshold": threshold,
        "threshold_settings": threshold_settings,
        "overall": overall,
        "split_report": split_report(
            {"train": train_rows, "validation": val_rows, "test": test_rows},
            split_strategy,
        ),
        "slices": slices,
        "worst_slices": worst_slices(slices),
    }


def worst_slices(slices: dict[str, dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_name, values in slices.items():
        for value, metrics in values.items():
            positive_f1 = metrics.get("positive_f1", metrics.get("f1"))
            if positive_f1 is None:
                continue
            rows.append(
                {
                    "field": field_name,
                    "value": value,
                    "positive_f1": float(positive_f1),
                    "negative_f1": float(metrics.get("negative_f1", 0.0)),
                    "macro_f1": float(metrics.get("macro_f1", 0.0)),
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "positive_rate": float(metrics.get("positive_rate", 0.0)),
                    "rows": int(metrics.get("n", 0)),
                }
            )
    return sorted(rows, key=lambda row: (row["positive_f1"], -row["rows"]))[:limit]
