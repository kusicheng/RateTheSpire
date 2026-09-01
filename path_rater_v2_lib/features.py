# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .data_io import *

def skill_bucket(record: dict[str, Any]) -> str:
    if record.get("skill_bucket"):
        return str(record["skill_bucket"])
    asc = float(record.get("ascension", 0) or 0)
    if asc >= 15:
        return "high"
    if asc >= 5:
        return "mid"
    return "low"


def item_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("id", value.get("name", "UNKNOWN")))
    return str(value)


def stable_bucket(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % HASH_BUCKETS


def validate_no_leakage(record: dict[str, Any], strict: bool = True) -> list[str]:
    """Return keys that look like non-decision-time data."""
    bad: list[str] = []
    for key in record:
        if key in KNOWN_DECISION_KEYS:
            continue
        key_text = str(key).lower()
        if any(re.search(pattern, key_text) for pattern in LEAK_PATTERNS):
            bad.append(str(key))
    if strict and bad:
        raise ValueError(f"Record includes possible leakage keys: {bad}")
    return bad


@dataclass
class FeatureConfig:
    numeric_fields: list[str] = field(default_factory=lambda: list(NUMERIC_FIELDS))
    categorical_values: dict[str, list[str]] = field(default_factory=dict)
    scaler: StandardScaler | None = None


def config_to_dict(config: FeatureConfig) -> dict[str, Any]:
    return {
        "numeric_fields": config.numeric_fields,
        "categorical_values": config.categorical_values,
        "scaler": config.scaler,
    }


def config_from_dict(data: dict[str, Any]) -> FeatureConfig:
    return FeatureConfig(
        numeric_fields=list(data["numeric_fields"]),
        categorical_values={key: list(value) for key, value in data["categorical_values"].items()},
        scaler=data["scaler"],
    )


def featurize_record(record: dict[str, Any], strict_leakage: bool = True) -> dict[str, Any]:
    validate_no_leakage(record, strict=strict_leakage)
    path = parse_path(record.get("candidate_path") or record.get("path"))
    counts = {room: path.count(room) for room in ROOM_TYPES}
    early_path = path[:5]
    first_three = path[:3]
    path_len = max(len(path), 1)
    distance_to_elite = next((idx + 1 for idx, room in enumerate(path) if room == "E"), 0)
    distance_to_rest = next((idx + 1 for idx, room in enumerate(path) if room == "R"), 0)
    distance_to_shop = next((idx + 1 for idx, room in enumerate(path) if room == "$"), 0)
    distance_to_boss = next((idx + 1 for idx, room in enumerate(path) if room == "BOSS"), len(path))
    rest_positions = [idx + 1 for idx, room in enumerate(path) if room == "R"]
    shop_positions = [idx + 1 for idx, room in enumerate(path) if room == "$"]
    last_rest = rest_positions[-1] if rest_positions else 0
    last_shop = shop_positions[-1] if shop_positions else 0
    elite_before_rest = 1.0 if distance_to_elite and (not distance_to_rest or distance_to_elite < distance_to_rest) else 0.0
    rest_after_elite = 1.0 if distance_to_elite and distance_to_rest and distance_to_rest > distance_to_elite else 0.0
    rest_before_elite = 1.0 if distance_to_rest and (not distance_to_elite or distance_to_rest < distance_to_elite) else 0.0
    shop_before_elite = 1.0 if distance_to_shop and (not distance_to_elite or distance_to_shop < distance_to_elite) else 0.0
    before_rest = path[: distance_to_rest - 1] if distance_to_rest else path
    after_last_rest = path[last_rest:] if last_rest else path
    hp = float(record.get("current_hp", record.get("hp", 0)) or 0)
    max_hp = float(record.get("max_hp", 0) or 0)
    gold = float(record.get("current_gold", record.get("gold", 0)) or 0)
    hp_ratio = hp / max_hp if max_hp > 0 else 0.0
    missing_hp_ratio = 1.0 - hp_ratio if max_hp > 0 else 0.0
    is_low_hp = 1.0 if max_hp > 0 and hp_ratio < 0.35 else 0.0
    is_mid_hp = 1.0 if 0.35 <= hp_ratio < 0.7 else 0.0
    is_high_hp = 1.0 if hp_ratio >= 0.7 else 0.0
    remaining_elites = float(counts.get("E", 0))
    remaining_shops = float(counts.get("$", 0))
    deck_size = len(record.get("deck") or [])
    relic_count = len(record.get("relics") or [])
    floor = float(record.get("floor", 0) or 0)
    floor_in_act = floor % 17.0
    act = float(record.get("act", 1) or 1)
    ascension = float(record.get("ascension", 0) or 0)
    path_len_value = float(len(path))
    is_long_path = 1.0 if path_len_value > 10 else 0.0
    late_act = 1.0 if floor_in_act >= 11.0 else 0.0
    elite_density = float(counts.get("E", 0)) / path_len
    monster_density = float(counts.get("M", 0)) / path_len
    rest_density = float(counts.get("R", 0)) / path_len
    shop_density = float(counts.get("$", 0)) / path_len
    danger_before_rest = float(before_rest.count("M") + 2 * before_rest.count("E"))
    danger_after_last_rest = float(after_last_rest.count("M") + 2 * after_last_rest.count("E"))
    early_danger = float(first_three.count("M") + 2 * first_three.count("E"))
    character = str(record.get("character", record.get("player_class", "UNKNOWN"))).upper()
    try:
        player_experience = max(float(record.get("player_experience", 0) or 0), 0.0)
    except (TypeError, ValueError):
        player_experience = 0.0

    numeric = {
        "ascension": ascension,
        "act": act,
        "floor": floor,
        "current_hp": hp,
        "max_hp": max_hp,
        "current_gold": gold,
        "deck_size": deck_size,
        "relic_count": relic_count,
        "potion_count": len(record.get("potions") or []),
        "path_len": path_len_value,
        "floor_in_act": floor_in_act,
        "act_progress": floor_in_act / 16.0,
        "path_len_ratio": path_len_value / MAX_PATH_LEN,
        "remaining_elites": remaining_elites,
        "remaining_rest_sites": float(counts.get("R", 0)),
        "remaining_shops": remaining_shops,
        "remaining_monsters": float(counts.get("M", 0)),
        "remaining_events": float(counts.get("?", 0)),
        "elite_density": elite_density,
        "monster_density": monster_density,
        "rest_density": rest_density,
        "shop_density": shop_density,
        "next_is_elite": 1.0 if path[:1] == ["E"] else 0.0,
        "next_is_rest": 1.0 if path[:1] == ["R"] else 0.0,
        "next_is_shop": 1.0 if path[:1] == ["$"] else 0.0,
        "early_elites": float(early_path.count("E")),
        "early_rest_sites": float(early_path.count("R")),
        "early_shops": float(early_path.count("$")),
        "first_three_elites": float(first_three.count("E")),
        "first_three_rest_sites": float(first_three.count("R")),
        "first_three_shops": float(first_three.count("$")),
        "first_three_monsters": float(first_three.count("M")),
        "first_three_events": float(first_three.count("?")),
        "distance_to_elite": float(distance_to_elite),
        "distance_to_rest": float(distance_to_rest),
        "distance_to_shop": float(distance_to_shop),
        "elite_before_rest": elite_before_rest,
        "rest_after_elite": rest_after_elite,
        "rest_before_elite": rest_before_elite,
        "shop_before_elite": shop_before_elite,
        "elites_before_rest": float(before_rest.count("E")),
        "monsters_before_rest": float(before_rest.count("M")),
        "events_before_rest": float(before_rest.count("?")),
        "low_hp_elite_pressure": 1.0 if is_low_hp and elite_before_rest else 0.0,
        "low_hp_rest_relief": 1.0 if is_low_hp and 0 < distance_to_rest <= 3 else 0.0,
        "high_gold_shop_access": 1.0 if gold >= 250 and 0 < distance_to_shop <= 4 else 0.0,
        "hp_ratio": hp_ratio,
        "missing_hp_ratio": missing_hp_ratio,
        "hp_per_remaining_elite": hp / max(remaining_elites, 1.0),
        "gold_per_remaining_shop": gold / max(remaining_shops, 1.0),
        "deck_per_remaining_elite": deck_size / max(remaining_elites, 1.0),
        "relics_per_remaining_elite": relic_count / max(remaining_elites, 1.0),
        "distance_to_boss": float(distance_to_boss),
        "last_rest_distance_to_boss": float(max(distance_to_boss - last_rest, 0)) if last_rest else float(distance_to_boss),
        "last_shop_distance_to_boss": float(max(distance_to_boss - last_shop, 0)) if last_shop else float(distance_to_boss),
        "elites_after_last_rest": float(after_last_rest.count("E")),
        "monsters_after_last_rest": float(after_last_rest.count("M")),
        "events_after_last_rest": float(after_last_rest.count("?")),
        "danger_before_rest": danger_before_rest,
        "danger_after_last_rest": danger_after_last_rest,
        "early_danger": early_danger,
        "path_len_x_act": path_len_value * act,
        "path_len_x_missing_hp": path_len_value * missing_hp_ratio,
        "path_len_x_low_hp": path_len_value * is_low_hp,
        "path_len_x_mid_hp": path_len_value * is_mid_hp,
        "path_len_x_elite_density": path_len_value * elite_density,
        "act_x_missing_hp": act * missing_hp_ratio,
        "act_x_elite_density": act * elite_density,
        "act_x_monster_density": act * monster_density,
        "ascension_x_elite_density": ascension * elite_density,
        "ascension_x_missing_hp": ascension * missing_hp_ratio,
        "low_hp_no_rest": 1.0 if is_low_hp and counts.get("R", 0) == 0 else 0.0,
        "low_hp_long_path": is_low_hp * is_long_path,
        "mid_hp_long_path": is_mid_hp * is_long_path,
        "high_hp_long_path": is_high_hp * is_long_path,
        "no_rest_long_path": 1.0 if is_long_path and counts.get("R", 0) == 0 else 0.0,
        "late_act_long_path": late_act * is_long_path,
        "late_act_low_hp": late_act * is_low_hp,
        "late_act_no_rest": 1.0 if late_act and counts.get("R", 0) == 0 else 0.0,
        "long_path_elite_pressure": is_long_path * remaining_elites * missing_hp_ratio,
        "long_path_monster_pressure": is_long_path * float(counts.get("M", 0)) * missing_hp_ratio,
        "silent_act2": 1.0 if character in {"THE_SILENT", "SILENT"} and act == 2 else 0.0,
        "silent_act3": 1.0 if character in {"THE_SILENT", "SILENT"} and act >= 3 else 0.0,
        "ironclad_act2": 1.0 if character == "IRONCLAD" and act == 2 else 0.0,
        "ironclad_act3": 1.0 if character == "IRONCLAD" and act >= 3 else 0.0,
        "gold_low": 1.0 if gold < 100 else 0.0,
        "gold_mid": 1.0 if 100 <= gold < 250 else 0.0,
        "gold_high": 1.0 if gold >= 250 else 0.0,
        "hp_low": is_low_hp,
        "hp_mid": is_mid_hp,
        "hp_high": is_high_hp,
        "player_prior_runs": float(record.get("player_prior_runs", 0) or 0),
        "player_prior_win_rate": float(record.get("player_prior_win_rate", 0) or 0),
        "player_prior_avg_floors": float(record.get("player_prior_avg_floors", 0) or 0),
        "player_experience_log": math.log1p(player_experience),
        "player_experience_zero": 1.0 if player_experience == 0 else 0.0,
    }
    item_hashes = [0.0] * HASH_BUCKETS
    for weight, key in [(1.0, "deck"), (2.0, "relics"), (1.5, "potions")]:
        for value in record.get(key) or []:
            item_hashes[stable_bucket(f"{key}:{item_id(value)}")] += weight
    for idx, value in enumerate(item_hashes):
        numeric[f"item_hash_{idx}"] = value

    room_ids = [ROOM_TO_ID.get(room, ROOM_TO_ID["UNKNOWN"]) for room in path[:MAX_PATH_LEN]]
    room_ids += [ROOM_TO_ID["PAD"]] * (MAX_PATH_LEN - len(room_ids))
    label = record.get("label", record.get("victory"))
    if isinstance(label, str):
        label = label.lower() in {"1", "true", "win", "victory", "yes"}
    floors_reached = record.get("floors_reached", record.get("floor_reached"))
    if floors_reached is None:
        floors_reached = MAX_EXPECTED_FLOOR if bool(label) else record.get("floor", 0)
    survival_target = max(0.0, min(float(floors_reached or 0), MAX_EXPECTED_FLOOR)) / MAX_EXPECTED_FLOOR
    return {
        "numeric": numeric,
        "rooms": room_ids,
        "categorical": {
            "character": str(record.get("character", record.get("player_class", "UNKNOWN"))),
            "skill_bucket": skill_bucket(record),
            "build_id": str(record.get("build_id", "UNKNOWN")),
            "game_mode": str(record.get("game_mode", "UNKNOWN")),
        },
        "label": None if label is None else int(bool(label)),
        "survival_target": survival_target,
    }
