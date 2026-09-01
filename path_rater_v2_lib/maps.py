# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .blend import blend_probability_arrays
from .data_io import normalize_room, parse_path, read_records
from .datasets import PathDecisionDataset, as_tabular_matrix
from .model_io import bundle_config, load_raw_bundle
from .neural import PathRaterNet

def node_identity(node: dict[str, Any], index: int) -> str:
    for key in ["id", "node_id", "key", "uuid"]:
        if key in node:
            return str(node[key])
    x_value = node.get("x", node.get("grid_x", node.get("col")))
    y_value = node.get("y", node.get("grid_y", node.get("row", node.get("floor"))))
    if x_value is not None and y_value is not None:
        return f"{x_value},{y_value}"
    return str(index)


def child_identity(value: Any) -> str:
    if isinstance(value, dict):
        if "target" in value:
            return child_identity(value["target"])
        if "to" in value:
            return child_identity(value["to"])
        return node_identity(value, 0)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{value[0]},{value[1]}"
    return str(value)


def map_nodes_from_save(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "saved_map" in data:
        return map_nodes_from_save(data["saved_map"])
    if isinstance(data, dict):
        for key in ["nodes", "map_points", "points", "map"]:
            if key in data:
                return map_nodes_from_save(data[key])
    if isinstance(data, list):
        nodes: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, list):
                nodes.extend(node for node in item if isinstance(node, dict))
            elif isinstance(item, dict):
                nodes.append(item)
        return nodes
    return []


def node_children(node: dict[str, Any]) -> list[str]:
    for key in ["children", "next_nodes", "connected_nodes", "edges", "links", "outgoing"]:
        if key in node:
            value = node[key] or []
            return [child_identity(child) for child in value]
    return []


def enumerate_candidate_paths_from_saved_map(
    saved_map: dict[str, Any],
    current_node: str | None = None,
    max_paths: int = 5000,
) -> list[list[str]]:
    nodes = map_nodes_from_save(saved_map)
    if not nodes:
        raise ValueError("No map nodes found in saved_map input.")

    node_by_id = {node_identity(node, idx): node for idx, node in enumerate(nodes)}
    children_by_id = {node_id: node_children(node) for node_id, node in node_by_id.items()}
    if current_node is None:
        referenced = {child for children in children_by_id.values() for child in children}
        starts = [node_id for node_id in node_by_id if node_id not in referenced]
    else:
        starts = [current_node]
    paths: list[list[str]] = []

    def visit(node_id: str, path: list[str], seen: set[str]) -> None:
        if len(paths) >= max_paths or node_id not in node_by_id or node_id in seen:
            return
        node = node_by_id[node_id]
        room = normalize_room(node.get("map_point_type", node.get("room_type", node.get("type", "UNKNOWN"))))
        new_path = path + [room]
        children = [child for child in children_by_id.get(node_id, []) if child in node_by_id]
        if not children or room == "BOSS":
            paths.append(new_path)
            return
        for child in children:
            visit(child, new_path, seen | {node_id})

    for start in starts:
        visit(start, [], set())
    return paths


def enumerate_saved_map_file(path: Path, current_node: str | None = None, max_paths: int = 5000) -> list[list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return enumerate_candidate_paths_from_saved_map(data, current_node=current_node, max_paths=max_paths)


def score_candidate_values(
    model_path: Path,
    state: dict[str, Any],
    candidates: list[Any],
) -> list[dict[str, Any]]:
    bundle = load_raw_bundle(model_path)
    return score_candidate_values_from_bundle(bundle, state, candidates)


def score_candidate_values_from_bundle(
    bundle: dict[str, Any],
    state: dict[str, Any],
    candidates: list[Any],
) -> list[dict[str, Any]]:
    config = bundle_config(bundle)
    rows = []
    for candidate in candidates:
        row = dict(state)
        if isinstance(candidate, dict):
            row["candidate_path"] = candidate.get("candidate_path", candidate.get("path", []))
        else:
            row["candidate_path"] = candidate
        row["label"] = 0
        rows.append(row)
    data = PathDecisionDataset(rows, config)
    if bundle.get("model_type") in {"hist_gradient_boosting", "extra_trees"}:
        matrix, _labels = as_tabular_matrix(data)
        probs = bundle["model"].predict_proba(matrix)[:, 1].tolist()
    elif bundle.get("model_type") == "hist_gradient_boosting_ensemble":
        matrix, _labels = as_tabular_matrix(data)
        prob_arrays = [model.predict_proba(matrix)[:, 1] for model in bundle["models"]]
        probs = blend_probability_arrays(prob_arrays, bundle.get("blend_weights", [])).tolist()
    else:
        model_args = bundle.get("model_args", {})
        model = PathRaterNet(
            numeric_dim=len(config.numeric_fields),
            cat_cardinalities=[len(config.categorical_values[field]) for field in CATEGORICAL_FIELDS],
            d_model=model_args.get("d_model", 64),
            use_position=model_args.get("use_position", False),
            use_attention_pooling=model_args.get("use_attention_pooling", False),
        )
        model.load_state_dict(bundle["model_state"], strict=False)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(data.numeric, data.rooms, data.cats)).numpy().tolist()
    result = []
    for candidate, prob in zip(candidates, probs):
        result.append({"candidate": candidate, "win_probability": float(prob)})
    return sorted(result, key=lambda item: item["win_probability"], reverse=True)


def score_candidates(model_path: Path, state_path: Path, candidates_path: Path) -> list[dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    return score_candidate_values(model_path, state, candidates)


def score_saved_map(
    model_path: Path,
    state_path: Path,
    map_path: Path,
    current_node: str | None,
    max_paths: int = 5000,
) -> list[dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    candidates = enumerate_saved_map_file(map_path, current_node=current_node, max_paths=max_paths)
    return score_candidate_values(model_path, state, candidates)


def load_json_value(value: Any, base_dir: Path | None = None) -> Any:
    if not isinstance(value, str):
        return value
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return value


def load_candidate_path_argument(value: Any, base_dir: Path | None = None) -> list[str]:
    if not isinstance(value, str):
        return parse_path(value)
    try:
        path = Path(value)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("candidate_path", data.get("path", []))
            return parse_path(data)
    except OSError:
        pass
    text = value.strip()
    if text.startswith("["):
        return parse_path(json.loads(text))
    return parse_path(text)


def rank_chosen_path(scored: list[dict[str, Any]], chosen_path: list[str]) -> dict[str, Any]:
    normalized_chosen = parse_path(chosen_path)
    for rank, item in enumerate(scored, start=1):
        candidate = item.get("candidate", item.get("path", []))
        if parse_path(candidate) == normalized_chosen:
            total = len(scored)
            top_score = float(scored[0]["win_probability"]) if scored else 0.0
            chosen_score = float(item["win_probability"])
            return {
                "chosen_path": normalized_chosen,
                "chosen_rank": rank,
                "candidate_count": total,
                "rank_percentile": 1.0 - ((rank - 1) / max(total - 1, 1)),
                "chosen_probability": chosen_score,
                "top_probability": top_score,
                "probability_gap_to_top": top_score - chosen_score,
                "top_candidate": scored[0]["candidate"] if scored else [],
                "chosen_found": True,
            }
    return {
        "chosen_path": normalized_chosen,
        "chosen_rank": None,
        "candidate_count": len(scored),
        "rank_percentile": None,
        "chosen_probability": None,
        "top_probability": float(scored[0]["win_probability"]) if scored else None,
        "probability_gap_to_top": None,
        "top_candidate": scored[0]["candidate"] if scored else [],
        "chosen_found": False,
    }


def evaluate_chosen_path_on_map(
    model_path: Path,
    state_path: Path,
    map_path: Path,
    chosen_path: list[str],
    current_node: str | None,
    max_paths: int = 5000,
) -> dict[str, Any]:
    scored = score_saved_map(
        model_path,
        state_path,
        map_path,
        current_node=current_node,
        max_paths=max_paths,
    )
    result = rank_chosen_path(scored, chosen_path)
    result["scored_candidates"] = scored
    return result


def summarize_choice_ranks(results: list[dict[str, Any]]) -> dict[str, Any]:
    found = [item for item in results if item.get("chosen_found")]
    total = len(results)
    found_count = len(found)
    return {
        "examples": total,
        "chosen_found": found_count,
        "chosen_missing": total - found_count,
        "top1_rate": float(sum(1 for item in found if item.get("chosen_rank") == 1) / found_count)
        if found_count
        else 0.0,
        "top3_rate": float(sum(1 for item in found if (item.get("chosen_rank") or 999999) <= 3) / found_count)
        if found_count
        else 0.0,
        "mean_reciprocal_rank": float(
            sum(1.0 / float(item["chosen_rank"]) for item in found if item.get("chosen_rank")) / found_count
        )
        if found_count
        else 0.0,
        "mean_rank_percentile": float(
            sum(float(item["rank_percentile"]) for item in found if item.get("rank_percentile") is not None)
            / found_count
        )
        if found_count
        else 0.0,
        "mean_probability_gap_to_top": float(
            sum(float(item["probability_gap_to_top"]) for item in found if item.get("probability_gap_to_top") is not None)
            / found_count
        )
        if found_count
        else 0.0,
    }


def evaluate_map_choice_records(
    model_path: Path,
    records: list[dict[str, Any]],
    base_dir: Path | None = None,
    max_paths: int = 5000,
    include_scored_candidates: bool = False,
) -> dict[str, Any]:
    bundle = load_raw_bundle(model_path)
    results = []
    for idx, record in enumerate(records):
        raw_state = record.get("state", record.get("state_path"))
        raw_map = record.get("map", record.get("map_path"))
        if raw_state is None or raw_map is None or "chosen_path" not in record:
            raise ValueError(f"Map-choice record {idx} must include state, map, and chosen_path.")
        state = load_json_value(raw_state, base_dir=base_dir)
        saved_map = load_json_value(raw_map, base_dir=base_dir)
        if not isinstance(state, dict):
            raise ValueError(f"Map-choice record {idx} state must resolve to a JSON object.")
        candidates = enumerate_candidate_paths_from_saved_map(
            saved_map,
            current_node=record.get("current_node"),
            max_paths=max_paths,
        )
        scored = score_candidate_values_from_bundle(bundle, state, candidates)
        result = rank_chosen_path(scored, load_candidate_path_argument(record["chosen_path"], base_dir=base_dir))
        result["example_index"] = idx
        if "run_id" in record:
            result["run_id"] = record["run_id"]
        if include_scored_candidates:
            result["scored_candidates"] = scored
        results.append(result)
    return {
        "summary": summarize_choice_ranks(results),
        "results": results,
    }
