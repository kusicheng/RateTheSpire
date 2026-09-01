# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *

def read_records(path: Path) -> list[dict[str, Any]]:
    """Read JSON, NDJSON, gzip NDJSON, or CSV records."""
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv"):
        return pd.read_csv(path).to_dict(orient="records")

    opener = gzip.open if suffixes.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        text = handle.read().strip()

    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of records.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_ndjson(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def normalize_room(room: Any) -> str:
    value = str(room).strip()
    mapping = {
        "MonsterRoom": "M",
        "EventRoom": "?",
        "TreasureRoom": "T",
        "RestRoom": "R",
        "ShopRoom": "$",
        "EliteRoom": "E",
        "BossRoom": "BOSS",
        "Enemy": "M",
        "Monster": "M",
        "monster": "M",
        "unknown": "?",
        "elite": "E",
        "rest_site": "R",
        "shop": "$",
        "treasure": "T",
        "ancient": "A",
        "boss": "BOSS",
        "B": "BOSS",
        "Unknown": "UNKNOWN",
    }
    return mapping.get(value, value if value in ROOM_TO_ID else "UNKNOWN")


def parse_path(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [normalize_room(v) for v in value]
    parts = [part for part in re.split(r"\s*(?:-|,|>)\s*", str(value)) if part]
    return [normalize_room(part) for part in parts]


def value_at_floor(record: dict[str, Any], keys: list[str], floor_idx: int, default: Any = 0) -> Any:
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, list):
            if floor_idx < len(value):
                return value[floor_idx]
            continue
        return value
    return default


def list_at_floor(record: dict[str, Any], keys: list[str], floor_idx: int) -> list[Any]:
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, list):
            if value and all(isinstance(item, list) for item in value):
                return value[min(floor_idx, len(value) - 1)]
            return value
    return []
