# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .features import FeatureConfig, config_from_dict
from .neural import PathRaterNet

def load_bundle(path: Path) -> tuple[PathRaterNet, FeatureConfig, dict[str, Any]]:
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    raw_config = bundle["config"]
    config = config_from_dict(raw_config) if isinstance(raw_config, dict) else raw_config
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
    return model, config, bundle.get("metrics", {})


def load_raw_bundle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def bundle_config(bundle: dict[str, Any]) -> FeatureConfig:
    raw_config = bundle["config"]
    return config_from_dict(raw_config) if isinstance(raw_config, dict) else raw_config
