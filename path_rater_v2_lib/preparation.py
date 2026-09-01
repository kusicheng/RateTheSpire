# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .data_io import *
from .features import item_id, skill_bucket, validate_no_leakage

def prepare_chosen_suffix_rows(records: Iterable[dict[str, Any]], max_floor: int | None = None) -> list[dict[str, Any]]:
    """Create decision rows from run records that contain the taken path."""
    prepared: list[dict[str, Any]] = []
    for record in records:
        full_path = parse_path(
            record.get("path_taken")
            or record.get("path")
            or record.get("path_per_floor")
            or record.get("room_path")
        )
        if not full_path:
            continue
        stop = min(len(full_path), max_floor or len(full_path))
        victory = record.get("victory", record.get("won", record.get("is_victory", False)))
        for idx in range(stop):
            prepared.append(
                {
                    "character": record.get("character", record.get("player_class", "UNKNOWN")),
                    "ascension": record.get("ascension", record.get("ascension_level", 0)),
                    "act": 1 + idx // 15,
                    "floor": idx,
                    "current_hp": value_at_floor(
                        record, ["current_hp_per_floor", "hp_per_floor", "current_hp", "hp"], idx
                    ),
                    "max_hp": value_at_floor(record, ["max_hp_per_floor", "max_hp"], idx),
                    "current_gold": value_at_floor(
                        record, ["current_gold_per_floor", "gold", "current_gold"], idx
                    ),
                    "deck": list_at_floor(record, ["deck_per_floor", "current_deck"], idx),
                    "relics": list_at_floor(record, ["relics_per_floor", "current_relics"], idx),
                    "potions": list_at_floor(record, ["potions_per_floor", "potions"], idx),
                    "candidate_path": sts1_visible_path(full_path, idx),
                    "floors_reached": record.get("floor_reached", record.get("floors_reached")),
                    "victory": victory,
                }
            )
    return prepared


def read_sts2_runs(path: Path) -> list[dict[str, Any]]:
    if path.is_file() and "".join(path.suffixes).lower() != ".run":
        return read_records(path)

    files = sorted(path.rglob("*.run")) if path.is_dir() else [path]
    records = []
    for file_path in files:
        try:
            record = json.loads(file_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            record = json.loads(file_path.read_text(encoding="utf-8-sig"))
        record["_run_file"] = str(file_path)
        records.append(record)
    return records


def read_sts1_runs(path: Path, max_files: int | None = None) -> list[dict[str, Any]]:
    files = sorted(path.rglob("*.json.gz")) + sorted(path.rglob("*.json")) if path.is_dir() else [path]
    if max_files is not None:
        files = files[:max_files]
    records: list[dict[str, Any]] = []
    for file_path in files:
        for raw in read_records(file_path):
            event = raw.get("event", raw) if isinstance(raw, dict) else raw
            if isinstance(event, dict):
                event["_run_file"] = str(file_path)
                records.append(event)
    return records


def ids_available_at(items: list[dict[str, Any]], current_floor: int) -> list[str]:
    ids = []
    for item in items or []:
        added = int(item.get("floor_added_to_deck", item.get("floor", 1)) or 1)
        if added <= current_floor:
            ids.append(str(item.get("id", "UNKNOWN")))
    return ids


def floor_of(item: dict[str, Any], default: int = 1) -> int:
    try:
        return int(float(item.get("floor", item.get("floor_added_to_deck", default)) or default))
    except (TypeError, ValueError):
        return default


def sts1_starting_deck(character: str, ascension: int) -> list[str]:
    deck_by_character = {
        "IRONCLAD": ["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
        "THE_SILENT": ["Strike_G"] * 5 + ["Defend_G"] * 5 + ["Survivor", "Neutralize"],
        "DEFECT": ["Strike_B"] * 4 + ["Defend_B"] * 4 + ["Zap", "Dualcast"],
        "WATCHER": ["Strike_P"] * 4 + ["Defend_P"] * 4 + ["Eruption", "Vigilance"],
    }
    deck = list(deck_by_character.get(str(character).upper(), []))
    if ascension >= 10:
        deck.insert(0, "AscendersBane")
    return deck


def sts1_starting_relics(character: str) -> list[str]:
    relic_by_character = {
        "IRONCLAD": "Burning Blood",
        "THE_SILENT": "Ring of the Snake",
        "DEFECT": "Cracked Core",
        "WATCHER": "PureWater",
    }
    relic = relic_by_character.get(str(character).upper())
    return [relic] if relic else []


def sts1_skill_bucket(record: dict[str, Any]) -> str:
    experience = record.get("player_experience")
    try:
        value = float(experience)
    except (TypeError, ValueError):
        value = -1.0
    if value >= 500000:
        return "high"
    if value >= 100000:
        return "mid"
    ascension = int(record.get("ascension_level", 0) or 0)
    if ascension >= 15:
        return "high"
    if ascension >= 5:
        return "mid"
    return "low"


def sts1_deck_at_floor(record: dict[str, Any], current_floor: int) -> list[str]:
    character = str(record.get("character_chosen", "UNKNOWN")).upper()
    ascension = int(record.get("ascension_level", 0) or 0)
    deck = sts1_starting_deck(character, ascension)
    for choice in record.get("card_choices") or []:
        if not isinstance(choice, dict) or floor_of(choice) > current_floor:
            continue
        picked = choice.get("picked")
        if picked and picked not in {"SKIP", "Singing Bowl"}:
            deck.append(str(picked))
    for event in record.get("event_choices") or []:
        if not isinstance(event, dict) or floor_of(event) > current_floor:
            continue
        for card in event.get("cards_obtained") or []:
            deck.append(str(card))
        removed = [str(card) for card in event.get("cards_removed") or []]
        transformed = [str(card) for card in event.get("cards_transformed") or []]
        for card in removed + transformed:
            if card in deck:
                deck.remove(card)
    for card, floor in zip(record.get("items_purged") or [], record.get("items_purged_floors") or []):
        try:
            purge_floor = int(float(floor))
        except (TypeError, ValueError):
            continue
        if purge_floor <= current_floor and str(card) in deck:
            deck.remove(str(card))
    return deck


def sts1_relics_at_floor(record: dict[str, Any], current_floor: int) -> list[str]:
    character = str(record.get("character_chosen", "UNKNOWN")).upper()
    relics = sts1_starting_relics(character)
    if current_floor >= 0:
        relics.append("NeowsBlessing")
        neow_bonus = record.get("neow_bonus")
        neow_cost = record.get("neow_cost")
        if neow_bonus:
            relics.append(f"NeowBonus:{neow_bonus}")
        if neow_cost:
            relics.append(f"NeowCost:{neow_cost}")
    for item in record.get("relics_obtained") or []:
        if isinstance(item, dict) and floor_of(item) <= current_floor:
            relics.append(str(item.get("key", "UNKNOWN")))
    for item in record.get("boss_relics") or []:
        if isinstance(item, dict) and floor_of(item, default=16) <= current_floor and item.get("picked"):
            relics.append(str(item["picked"]))
    return relics


def sts1_potions_at_floor(record: dict[str, Any], current_floor: int) -> list[str]:
    potions = [
        str(item.get("key", "UNKNOWN"))
        for item in record.get("potions_obtained") or []
        if isinstance(item, dict) and floor_of(item) <= current_floor
    ]
    used = sum(1 for floor in record.get("potions_floor_usage") or [] if int(float(floor)) <= current_floor)
    return potions[used:]


def sts1_path_rooms(record: dict[str, Any]) -> list[str | None]:
    raw_path = record.get("path_per_floor") or record.get("path_taken")
    if isinstance(raw_path, list):
        return [None if room is None else normalize_room(room) for room in raw_path]
    return parse_path(raw_path)


def sts1_visible_path(path: list[str | None], start_idx: int) -> list[str]:
    visible_path: list[str] = []
    for room in path[start_idx:]:
        if room is None or room == "UNKNOWN":
            continue
        visible_path.append(room)
        if room == "BOSS":
            break
    return visible_path


def sts1_label(record: dict[str, Any], label_mode: str, current_floor: int, visible_path: list[str]) -> bool:
    if label_mode == "victory":
        return bool(record.get("victory", False))
    if label_mode == "act-survival":
        floor_reached = int(record.get("floor_reached", 0) or 0)
        return floor_reached > current_floor + len(visible_path)
    raise ValueError(f"Unknown STS1 label mode: {label_mode}")


def has_sane_state(row: dict[str, Any]) -> bool:
    hp = float(row.get("current_hp", 0) or 0)
    max_hp = float(row.get("max_hp", 0) or 0)
    gold = float(row.get("current_gold", 0) or 0)
    if hp < 0 or max_hp <= 0 or hp > max_hp:
        return False
    if max_hp > 300:
        return False
    if gold < 0 or gold > 5000:
        return False
    return True


def sts1_run_to_decisions(
    record: dict[str, Any],
    max_floor: int | None = None,
    label_mode: str = "victory",
    include_invalid_state: bool = False,
) -> list[dict[str, Any]]:
    """Create no-leakage chosen-suffix rows from one STS1 run."""
    full_path = sts1_path_rooms(record)
    if not full_path:
        return []
    stop = min(len(full_path), max_floor or len(full_path))
    run_id = str(record.get("play_id") or record.get("seed_played") or record.get("_run_file") or "")
    rows: list[dict[str, Any]] = []
    for idx in range(stop):
        if full_path[idx] is None or full_path[idx] == "UNKNOWN":
            continue
        visible_path = sts1_visible_path(full_path, idx)
        if not visible_path:
            continue
        row = {
            "run_id": run_id,
            "build_id": record.get("build_version", "UNKNOWN"),
            "game_mode": "standard",
            "profile_type": "vanilla",
            "player_count": 1,
            "was_abandoned": False,
            "is_cheated": False,
            "player_prior_runs": 0.0,
            "player_prior_win_rate": 0.0,
            "player_prior_avg_floors": 0.0,
            "player_experience": record.get("player_experience", 0),
            "character": record.get("character_chosen", "UNKNOWN"),
            "skill_bucket": sts1_skill_bucket(record),
            "ascension": record.get("ascension_level", 0),
            "act": 1 + idx // 17,
            "floor": idx,
            "current_hp": value_at_floor(record, ["current_hp_per_floor"], idx),
            "max_hp": value_at_floor(record, ["max_hp_per_floor"], idx),
            "current_gold": value_at_floor(record, ["gold_per_floor", "gold"], idx),
            "deck": sts1_deck_at_floor(record, idx),
            "relics": sts1_relics_at_floor(record, idx),
            "potions": sts1_potions_at_floor(record, idx),
            "candidate_path": visible_path,
            "floors_reached": record.get("floor_reached", len(full_path)),
            "victory": sts1_label(record, label_mode, idx, visible_path),
        }
        if include_invalid_state or has_sane_state(row):
            rows.append(row)
    return rows


def prepare_sts1_run_history(
    records: Iterable[dict[str, Any]],
    max_rows: int | None = None,
    max_floor: int | None = None,
    label_mode: str = "victory",
    include_daily: bool = False,
    include_trial: bool = False,
    include_endless: bool = False,
    characters: set[str] | None = None,
    include_invalid_state: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("is_daily") and not include_daily:
            continue
        if record.get("is_trial") and not include_trial:
            continue
        if record.get("is_endless") and not include_endless:
            continue
        if characters and str(record.get("character_chosen")) not in characters:
            continue
        rows.extend(
            sts1_run_to_decisions(
                record,
                max_floor=max_floor,
                label_mode=label_mode,
                include_invalid_state=include_invalid_state,
            )
        )
        if max_rows and len(rows) >= max_rows:
            return rows[:max_rows]
    return rows


def first_player(record: dict[str, Any]) -> dict[str, Any]:
    players = record.get("players") or []
    return players[0] if players else {}


def first_player_stats(point: dict[str, Any]) -> dict[str, Any]:
    stats = point.get("player_stats") or []
    return stats[0] if stats else {}


def flatten_sts2_points(record: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    points: list[tuple[int, dict[str, Any]]] = []
    for act_idx, act_points in enumerate(record.get("map_point_history") or []):
        for point in act_points or []:
            points.append((act_idx + 1, point))
    return points


def profile_type(record: dict[str, Any]) -> str:
    return "modded" if "modded" in str(record.get("_run_file", "")).lower() else "vanilla"


def run_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_type": profile_type(record),
        "player_count": len(record.get("players") or []),
        "was_abandoned": bool(record.get("was_abandoned", False)),
        "is_cheated": bool(record.get("_isCheated", False)),
    }


def player_identity(record: dict[str, Any]) -> str | None:
    for key in ("player_id", "profile_id", "user_id", "_userId", "steam_id"):
        if record.get(key) is not None:
            return str(record[key])
    if record.get("_run_file"):
        text = str(record["_run_file"]).replace("\\", "/").lower()
        marker = "/saves/history/"
        if marker in text:
            return text.split(marker, 1)[0]
        return str(Path(str(record["_run_file"])).parent)
    if record.get("_serverId") is not None:
        return None
    return "single_input"


def sts2_run_to_decisions(
    record: dict[str, Any],
    include_start: bool = True,
    skill_features: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    acts = record.get("map_point_history") or []
    points = flatten_sts2_points(record)
    if not points:
        return []

    player = first_player(record)
    run_id = str(record.get("start_time") or record.get("seed") or record.get("_run_file") or "")
    skill_features = skill_features or {}
    metadata = run_metadata(record)
    floors_reached = len(points)
    rows: list[dict[str, Any]] = []
    first_stats = first_player_stats(points[0][1])
    first_act_path = [normalize_room(point.get("map_point_type")) for point in (acts[0] or [])]

    if include_start:
        rows.append(
            {
                "run_id": run_id,
                "build_id": record.get("build_id", "UNKNOWN"),
                "game_mode": record.get("game_mode", "UNKNOWN"),
                **metadata,
                "player_prior_runs": skill_features.get("player_prior_runs", 0.0),
                "player_prior_win_rate": skill_features.get("player_prior_win_rate", 0.0),
                "player_prior_avg_floors": skill_features.get("player_prior_avg_floors", 0.0),
                "character": player.get("character", "UNKNOWN"),
                "ascension": record.get("ascension", 0),
                "act": 1,
                "floor": 0,
                "current_hp": first_stats.get("max_hp", player.get("max_hp", 0)),
                "max_hp": first_stats.get("max_hp", player.get("max_hp", 0)),
                "current_gold": 0,
                "deck": ids_available_at(player.get("deck") or [], 1),
                "relics": ids_available_at(player.get("relics") or [], 1),
                "potions": [str(item.get("id", item)) for item in player.get("potions") or []],
                "candidate_path": first_act_path,
                "floors_reached": floors_reached,
                "victory": record.get("win", False),
            }
        )

    current_floor = 0
    for act_idx, act_points in enumerate(acts):
        act_path = [normalize_room(point.get("map_point_type")) for point in (act_points or [])]
        for idx, point in enumerate(act_points or []):
            current_floor += 1
            remaining_path = act_path[idx + 1 :]
            if not remaining_path:
                continue
            stats = first_player_stats(point)
            rows.append(
                {
                    "run_id": run_id,
                    "build_id": record.get("build_id", "UNKNOWN"),
                    "game_mode": record.get("game_mode", "UNKNOWN"),
                    **metadata,
                    "player_prior_runs": skill_features.get("player_prior_runs", 0.0),
                    "player_prior_win_rate": skill_features.get("player_prior_win_rate", 0.0),
                    "player_prior_avg_floors": skill_features.get("player_prior_avg_floors", 0.0),
                    "character": player.get("character", "UNKNOWN"),
                    "ascension": record.get("ascension", 0),
                    "act": act_idx + 1,
                    "floor": current_floor,
                    "current_hp": stats.get("current_hp", 0),
                    "max_hp": stats.get("max_hp", player.get("max_hp", 0)),
                    "current_gold": stats.get("current_gold", 0),
                    "deck": ids_available_at(player.get("deck") or [], current_floor),
                    "relics": ids_available_at(player.get("relics") or [], current_floor),
                    "potions": [str(item.get("id", item)) for item in player.get("potions") or []],
                    "candidate_path": remaining_path,
                    "floors_reached": floors_reached,
                    "victory": record.get("win", False),
                }
            )
    return rows


def prepare_sts2_run_history(
    records: Iterable[dict[str, Any]],
    max_rows: int | None = None,
    include_abandoned: bool = False,
    build_ids: set[str] | None = None,
    characters: set[str] | None = None,
    game_modes: set[str] | None = None,
    vanilla_only: bool = True,
    include_coop: bool = False,
    include_cheated: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior_by_player: dict[str, dict[str, float]] = {}
    sorted_records = sorted(records, key=lambda item: int(item.get("start_time", 0) or 0))
    for record in sorted_records:
        if record.get("was_abandoned") and not include_abandoned:
            continue
        if record.get("_isCheated") and not include_cheated:
            continue
        if vanilla_only and "modded" in str(record.get("_run_file", "")).lower():
            continue
        if not include_coop and (len(record.get("players") or []) != 1 or "coop" in str(record.get("game_mode", "")).lower()):
            continue
        if build_ids and str(record.get("build_id")) not in build_ids:
            continue
        if game_modes and str(record.get("game_mode")) not in game_modes:
            continue
        player = first_player(record)
        if characters and str(player.get("character")) not in characters:
            continue
        identity = player_identity(record)
        prior = prior_by_player.get(identity or "", {"runs": 0.0, "wins": 0.0, "floors": 0.0})
        skill_features = {
            "player_prior_runs": float(prior["runs"]) if identity else 0.0,
            "player_prior_win_rate": float(prior["wins"] / prior["runs"]) if identity and prior["runs"] else 0.0,
            "player_prior_avg_floors": float(prior["floors"] / prior["runs"]) if identity and prior["runs"] else 0.0,
        }
        rows.extend(sts2_run_to_decisions(record, skill_features=skill_features))
        if identity:
            prior["runs"] += 1.0
            prior["wins"] += float(bool(record.get("win", False)))
            prior["floors"] += float(len(flatten_sts2_points(record)))
            prior_by_player[identity] = prior
        if max_rows and len(rows) >= max_rows:
            return rows[:max_rows]
    return rows


def audit_decision_rows(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    runs = {record.get("run_id") for record in records if record.get("run_id") is not None}
    labels = [int(bool(record.get("label", record.get("victory")))) for record in records]
    profile_counts = pd.Series([record.get("profile_type", "UNKNOWN") for record in records]).value_counts().to_dict()
    game_mode_counts = pd.Series([record.get("game_mode", "UNKNOWN") for record in records]).value_counts().to_dict()
    player_count_counts = pd.Series([record.get("player_count", "UNKNOWN") for record in records]).value_counts().to_dict()
    cheated_counts = pd.Series([record.get("is_cheated", False) for record in records]).value_counts().to_dict()
    build_counts = pd.Series([record.get("build_id", "UNKNOWN") for record in records]).value_counts().to_dict()
    character_counts = pd.Series([record.get("character", "UNKNOWN") for record in records]).value_counts().to_dict()
    leakage_keys = sorted({key for record in records for key in validate_no_leakage(record, strict=False)})
    return {
        "rows": len(records),
        "runs": len(runs),
        "positive_rows": int(sum(labels)),
        "negative_rows": int(len(labels) - sum(labels)),
        "positive_rate": float(sum(labels) / len(labels)),
        "profile_type_counts": profile_counts,
        "game_mode_counts": game_mode_counts,
        "player_count_counts": player_count_counts,
        "cheated_counts": cheated_counts,
        "build_counts": build_counts,
        "character_counts": character_counts,
        "possible_leakage_keys": leakage_keys,
        "valid_main_population": profile_counts.get("modded", 0) == 0
        and set(str(key).lower() for key in game_mode_counts) <= {"standard"}
        and player_count_counts.get(1, player_count_counts.get("1", 0)) == len(records)
        and cheated_counts.get(True, cheated_counts.get("True", 0)) == 0,
    }


def last_history_stats(data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    points = flatten_sts2_points(data)
    if not points:
        return 0, {}
    return len(points), first_player_stats(points[-1][1])


def extract_state_from_current_run(data: dict[str, Any]) -> dict[str, Any]:
    player = first_player(data)
    floor, stats = last_history_stats(data)
    act = int(data.get("current_act_index", 0) or 0) + 1
    current_hp = stats.get("current_hp", player.get("current_hp", player.get("hp", 0)))
    max_hp = stats.get("max_hp", player.get("max_hp", 0))
    current_gold = stats.get("current_gold", player.get("current_gold", player.get("gold", 0)))
    current_floor = max(1, floor)
    return {
        "build_id": data.get("build_id", "UNKNOWN"),
        "game_mode": data.get("game_mode", "standard"),
        "character": player.get("character", "UNKNOWN"),
        "ascension": data.get("ascension", 0),
        "act": act,
        "floor": floor,
        "current_hp": current_hp,
        "max_hp": max_hp,
        "current_gold": current_gold,
        "deck": ids_available_at(player.get("deck") or [], current_floor),
        "relics": ids_available_at(player.get("relics") or [], current_floor),
        "potions": [item_id(item) for item in player.get("potions") or []],
        "skill_bucket": skill_bucket(data),
    }


def extract_state_file(path: Path) -> dict[str, Any]:
    return extract_state_from_current_run(json.loads(path.read_text(encoding="utf-8")))
