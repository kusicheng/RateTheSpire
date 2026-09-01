# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *

def parse_hgb_ensemble_configs(config_text: str | None) -> list[dict[str, float | int]]:
    if not config_text:
        return [
            {"learning_rate": 0.03, "max_iter": 800, "max_leaf_nodes": 63, "l2_regularization": 0.0},
            {"learning_rate": 0.05, "max_iter": 600, "max_leaf_nodes": 31, "l2_regularization": 0.0},
            {"learning_rate": 0.08, "max_iter": 500, "max_leaf_nodes": 63, "l2_regularization": 0.01},
        ]
    configs: list[dict[str, float | int]] = []
    for raw_config in config_text.split(","):
        parts = [part.strip() for part in raw_config.split(":")]
        if len(parts) != 4:
            raise ValueError("Each HGB ensemble config must be learning_rate:max_iter:max_leaf_nodes:l2_regularization.")
        configs.append(
            {
                "learning_rate": float(parts[0]),
                "max_iter": int(parts[1]),
                "max_leaf_nodes": int(parts[2]),
                "l2_regularization": float(parts[3]),
            }
        )
    return configs


def equal_blend_weights(count: int) -> list[float]:
    if count <= 0:
        return []
    return [1.0 / count] * count


def simplex_weight_grid(count: int, step: float = 0.05) -> list[list[float]]:
    if count <= 0:
        return []
    units = int(round(1.0 / step))
    if count == 1:
        return [[1.0]]
    weights: list[list[float]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append([(value / units) for value in [*prefix, remaining]])
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots - 1)

    visit([], units, count)
    return weights


def blend_probability_arrays(prob_arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if not prob_arrays:
        return np.array([], dtype=np.float32)
    if not weights:
        weights = equal_blend_weights(len(prob_arrays))
    return np.average(np.vstack(prob_arrays), axis=0, weights=np.array(weights, dtype=np.float64))


def select_ensemble_blend_weights(
    prob_arrays: list[np.ndarray],
    labels: list[int],
    mode: str = "equal",
    step: float = 0.05,
) -> list[float]:
    from .metrics import best_threshold_from_probs

    if mode == "equal":
        return equal_blend_weights(len(prob_arrays))
    if mode != "validation":
        raise ValueError(f"Unknown blend mode: {mode}")
    best_weights = equal_blend_weights(len(prob_arrays))
    best_score = -1.0
    for weights in simplex_weight_grid(len(prob_arrays), step=step):
        probs = blend_probability_arrays(prob_arrays, weights).tolist()
        threshold = best_threshold_from_probs(probs, labels)
        preds = [1 if prob >= threshold else 0 for prob in probs]
        score = f1_score(labels, preds, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_weights = weights
    return best_weights
