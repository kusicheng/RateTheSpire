# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .features import FeatureConfig, featurize_record

class PathDecisionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], config: FeatureConfig, fit: bool = False):
        self.config = config
        features = [featurize_record(row) for row in rows]
        if not features:
            raise ValueError("No training rows were provided.")

        for field_name in CATEGORICAL_FIELDS:
            if fit:
                values = sorted({item["categorical"][field_name] for item in features} | {"UNKNOWN"})
                config.categorical_values[field_name] = values
            elif field_name not in config.categorical_values:
                raise ValueError(f"Missing categorical config for {field_name}.")

        numeric = np.array(
            [[item["numeric"].get(field, 0.0) for field in config.numeric_fields] for item in features],
            dtype=np.float32,
        )
        if fit:
            config.scaler = StandardScaler()
            numeric = config.scaler.fit_transform(numeric).astype(np.float32)
        elif config.scaler is not None:
            numeric = config.scaler.transform(numeric).astype(np.float32)

        cats: list[list[int]] = []
        for item in features:
            cat_row = []
            for field_name in CATEGORICAL_FIELDS:
                values = config.categorical_values[field_name]
                value = item["categorical"][field_name]
                cat_row.append(values.index(value) if value in values else values.index("UNKNOWN"))
            cats.append(cat_row)

        labels = [item["label"] for item in features]
        if any(label is None for label in labels):
            raise ValueError("All training rows must include label or victory.")

        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.rooms = torch.tensor([item["rooms"] for item in features], dtype=torch.long)
        self.cats = torch.tensor(cats, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.survival_targets = torch.tensor([item["survival_target"] for item in features], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.numeric[idx], self.rooms[idx], self.cats[idx], self.labels[idx], self.survival_targets[idx]


def as_tabular_matrix(data: PathDecisionDataset) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.hstack(
        [
            data.numeric.numpy(),
            data.rooms.numpy().astype(np.float32),
            data.cats.numpy().astype(np.float32),
        ]
    )
    return matrix, data.labels.numpy().astype(int)
