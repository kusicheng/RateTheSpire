# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *

def split_records(
    records: list[dict[str, Any]],
    test_size: float,
    val_size: float,
    seed: int,
    strategy: str = "stratified-group",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups = [record.get("run_id") for record in records]
    if all(group is not None for group in groups) and len(set(groups)) >= 5:
        by_group: dict[str, list[dict[str, Any]]] = {}
        group_labels: dict[str, int] = {}
        for record in records:
            group = str(record["run_id"])
            by_group.setdefault(group, []).append(record)
            group_labels[group] = int(bool(record.get("label", record.get("victory"))))
        group_ids = sorted(by_group)
        if strategy == "chronological":
            group_ids = sorted(group_ids, key=lambda group: int(group) if str(group).isdigit() else str(group))
            test_count = max(1, int(round(len(group_ids) * test_size)))
            val_count = max(1, int(round((len(group_ids) - test_count) * val_size)))
            train_ids = group_ids[: len(group_ids) - test_count - val_count]
            val_ids = group_ids[len(group_ids) - test_count - val_count : len(group_ids) - test_count]
            test_ids = group_ids[len(group_ids) - test_count :]

            def chronological_rows_for(ids: list[str]) -> list[dict[str, Any]]:
                return [record for group in ids for record in by_group[group]]

            return chronological_rows_for(train_ids), chronological_rows_for(val_ids), chronological_rows_for(test_ids)

        if strategy != "stratified-group":
            raise ValueError(f"Unknown split strategy: {strategy}")
        labels = [group_labels[group] for group in group_ids]
        stratify = labels if min(labels.count(0), labels.count(1)) >= 2 else None
        train_val_ids, test_ids = train_test_split(
            group_ids,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        train_val_labels = [group_labels[group] for group in train_val_ids]
        stratify_val = train_val_labels if min(train_val_labels.count(0), train_val_labels.count(1)) >= 2 else None
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size,
            random_state=seed,
            stratify=stratify_val,
        )

        def rows_for(ids: list[str]) -> list[dict[str, Any]]:
            return [record for group in ids for record in by_group[group]]

        return rows_for(train_ids), rows_for(val_ids), rows_for(test_ids)

    labels = [int(bool(record.get("label", record.get("victory")))) for record in records]
    train_rows, test_rows = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_labels = [int(bool(record.get("label", record.get("victory")))) for record in train_rows]
    train_rows, val_rows = train_test_split(
        train_rows,
        test_size=val_size,
        random_state=seed,
        stratify=train_labels,
    )
    return train_rows, val_rows, test_rows


def row_label(record: dict[str, Any]) -> int:
    return int(bool(record.get("label", record.get("victory"))))


def split_report(rows_by_name: dict[str, list[dict[str, Any]]], strategy: str) -> dict[str, Any]:
    groups_by_name = {
        name: {str(row.get("run_id", idx)) for idx, row in enumerate(rows)}
        for name, rows in rows_by_name.items()
    }
    group_overlaps: dict[str, int] = {}
    names = list(groups_by_name)
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            group_overlaps[f"{left}_{right}"] = len(groups_by_name[left] & groups_by_name[right])

    splits: dict[str, Any] = {}
    for name, rows in rows_by_name.items():
        positives = sum(row_label(row) for row in rows)
        splits[name] = {
            "rows": len(rows),
            "runs": len(groups_by_name[name]),
            "positive_rows": positives,
            "negative_rows": len(rows) - positives,
            "positive_rate": float(positives / len(rows)) if rows else 0.0,
        }

    return {
        "split_strategy": strategy,
        "splits": splits,
        "run_group_overlaps": group_overlaps,
    }


def sample_weights_for_rows(rows: list[dict[str, Any]], mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode not in {"inverse-run", "balanced", "balanced-inverse-run"}:
        raise ValueError(f"Unknown sample weight mode: {mode}")
    weights = np.ones(len(rows), dtype=np.float32)
    if mode in {"inverse-run", "balanced-inverse-run"}:
        counts: dict[str, int] = {}
        for idx, row in enumerate(rows):
            run_id = str(row.get("run_id", idx))
            counts[run_id] = counts.get(run_id, 0) + 1
        weights = np.array([1.0 / counts[str(row.get("run_id", idx))] for idx, row in enumerate(rows)], dtype=np.float32)
    if mode in {"balanced", "balanced-inverse-run"}:
        labels = [row_label(row) for row in rows]
        class_sums = {
            label: float(sum(weight for weight, row_label_value in zip(weights, labels) if row_label_value == label))
            for label in {0, 1}
        }
        total = float(weights.sum())
        for idx, label in enumerate(labels):
            if class_sums.get(label, 0.0) > 0:
                weights[idx] *= total / (2.0 * class_sums[label])
    return weights
