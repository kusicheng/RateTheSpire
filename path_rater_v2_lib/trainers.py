# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .blend import *
from .data_io import read_records
from .datasets import PathDecisionDataset, as_tabular_matrix
from .features import FeatureConfig, config_to_dict
from .metrics import *
from .model_io import bundle_config, load_raw_bundle
from .neural import PathRaterNet, resolve_device
from .splits import row_label, sample_weights_for_rows, split_records, split_report

def train_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = read_records(Path(args.input))
    if args.max_rows:
        records = records[: args.max_rows]
    if len(records) < 20:
        raise ValueError("Need at least 20 labeled decision rows for training.")

    train_rows, val_rows, test_rows = split_records(
        records,
        args.test_size,
        args.val_size,
        args.seed,
        strategy=args.split_strategy,
    )

    config = FeatureConfig()
    train_data = PathDecisionDataset(train_rows, config, fit=True)
    val_data = PathDecisionDataset(val_rows, config)
    test_data = PathDecisionDataset(test_rows, config)
    device = resolve_device(args.device)

    model = PathRaterNet(
        numeric_dim=len(config.numeric_fields),
        cat_cardinalities=[len(config.categorical_values[field]) for field in CATEGORICAL_FIELDS],
        d_model=args.d_model,
        use_position=not args.no_position,
        use_attention_pooling=not args.no_attention_pooling,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = None
    raw_pos_weight = str(args.pos_weight).strip().lower()
    if raw_pos_weight == "auto":
        positives = float(train_data.labels.sum().item())
        negatives = float(len(train_data.labels) - positives)
        if positives > 0:
            pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    elif raw_pos_weight not in {"", "0", "none", "false"}:
        pos_weight = torch.tensor([float(args.pos_weight)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    aux_loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    best_state = None
    best_f1 = -1.0
    patience_left = args.patience
    for _epoch in range(args.epochs):
        model.train()
        for numeric, rooms, cats, labels, survival_targets in train_loader:
            numeric = numeric.to(device)
            rooms = rooms.to(device)
            cats = cats.to(device)
            labels = labels.to(device)
            survival_targets = survival_targets.to(device)
            optimizer.zero_grad()
            logits, floor_pred = model(numeric, rooms, cats, return_aux=True)
            loss = loss_fn(logits, labels) + args.aux_floor_weight * aux_loss_fn(floor_pred, survival_targets)
            loss.backward()
            optimizer.step()

        val_f1 = evaluate_model(model, val_loader)["f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    threshold = best_threshold(model, DataLoader(val_data, batch_size=args.batch_size))
    metrics = evaluate_model(model, DataLoader(test_data, batch_size=args.batch_size), threshold=threshold)
    metrics["threshold"] = threshold
    metrics["device"] = str(device)
    metrics["split_report"] = split_report(
        {"train": train_rows, "validation": val_rows, "test": test_rows},
        args.split_strategy,
    )
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    bundle = {
        "model_state": model_state,
        "config": config_to_dict(config),
        "metrics": metrics,
        "model_args": {
            "d_model": args.d_model,
            "aux_floor_weight": args.aux_floor_weight,
            "pos_weight": args.pos_weight,
            "use_position": not args.no_position,
            "use_attention_pooling": not args.no_attention_pooling,
        },
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as handle:
        pickle.dump(bundle, handle)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_tabular_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    records = read_records(Path(args.input))
    if args.max_rows:
        records = records[: args.max_rows]
    if len(records) < 20:
        raise ValueError("Need at least 20 labeled decision rows for training.")

    train_rows, val_rows, test_rows = split_records(
        records,
        args.test_size,
        args.val_size,
        args.seed,
        strategy=args.split_strategy,
    )
    config = FeatureConfig()
    train_data = PathDecisionDataset(train_rows, config, fit=True)
    val_data = PathDecisionDataset(val_rows, config)
    test_data = PathDecisionDataset(test_rows, config)
    x_train, y_train = as_tabular_matrix(train_data)
    x_val, y_val = as_tabular_matrix(val_data)
    x_test, y_test = as_tabular_matrix(test_data)

    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        random_state=args.seed,
    )
    sample_weight = sample_weights_for_rows(train_rows, args.sample_weight)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    val_probs = model.predict_proba(x_val)[:, 1]
    threshold_settings = make_threshold_settings(
        args.threshold_mode,
        val_probs.tolist(),
        val_rows,
        y_val.tolist(),
        min_rows=args.threshold_min_rows,
    )
    threshold = float(threshold_settings.get("fallback_threshold", threshold_settings.get("threshold", 0.5)))
    test_probs = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_probabilities_with_threshold_settings(
        test_probs.tolist(),
        test_rows,
        y_test.tolist(),
        threshold_settings,
    )
    metrics["threshold"] = threshold
    metrics["threshold_mode"] = args.threshold_mode
    metrics["threshold_settings"] = threshold_settings
    metrics["model_type"] = "hist_gradient_boosting"
    metrics["sample_weight"] = args.sample_weight
    metrics["split_report"] = split_report(
        {"train": train_rows, "validation": val_rows, "test": test_rows},
        args.split_strategy,
    )

    bundle = {
        "model_type": "hist_gradient_boosting",
        "model": model,
        "config": config_to_dict(config),
        "metrics": metrics,
        "threshold_settings": threshold_settings,
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as handle:
        pickle.dump(bundle, handle)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_extra_trees_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    records = read_records(Path(args.input))
    if args.max_rows:
        records = records[: args.max_rows]
    if len(records) < 20:
        raise ValueError("Need at least 20 labeled decision rows for training.")

    train_rows, val_rows, test_rows = split_records(
        records,
        args.test_size,
        args.val_size,
        args.seed,
        strategy=args.split_strategy,
    )
    config = FeatureConfig()
    train_data = PathDecisionDataset(train_rows, config, fit=True)
    val_data = PathDecisionDataset(val_rows, config)
    test_data = PathDecisionDataset(test_rows, config)
    x_train, y_train = as_tabular_matrix(train_data)
    x_val, y_val = as_tabular_matrix(val_data)
    x_test, y_test = as_tabular_matrix(test_data)

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=args.class_weight,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    sample_weight = sample_weights_for_rows(train_rows, args.sample_weight)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    val_probs = model.predict_proba(x_val)[:, 1]
    threshold_settings = make_threshold_settings(
        args.threshold_mode,
        val_probs.tolist(),
        val_rows,
        y_val.tolist(),
        min_rows=args.threshold_min_rows,
    )
    threshold = float(threshold_settings.get("fallback_threshold", threshold_settings.get("threshold", 0.5)))
    test_probs = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_probabilities_with_threshold_settings(
        test_probs.tolist(),
        test_rows,
        y_test.tolist(),
        threshold_settings,
    )
    metrics["threshold"] = threshold
    metrics["threshold_mode"] = args.threshold_mode
    metrics["threshold_settings"] = threshold_settings
    metrics["model_type"] = "extra_trees"
    metrics["n_estimators"] = args.n_estimators
    metrics["max_depth"] = args.max_depth
    metrics["min_samples_leaf"] = args.min_samples_leaf
    metrics["max_features"] = args.max_features
    metrics["class_weight"] = args.class_weight
    metrics["sample_weight"] = args.sample_weight
    metrics["split_report"] = split_report(
        {"train": train_rows, "validation": val_rows, "test": test_rows},
        args.split_strategy,
    )

    bundle = {
        "model_type": "extra_trees",
        "model": model,
        "config": config_to_dict(config),
        "metrics": metrics,
        "threshold_settings": threshold_settings,
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as handle:
        pickle.dump(bundle, handle)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_model_blend(args: argparse.Namespace) -> dict[str, Any]:
    records = read_records(Path(args.input))
    if args.max_rows:
        records = records[: args.max_rows]
    if len(records) < 20:
        raise ValueError("Need at least 20 labeled decision rows for blend evaluation.")
    train_rows, val_rows, test_rows = split_records(
        records,
        args.test_size,
        args.val_size,
        args.seed,
        strategy=args.split_strategy,
    )
    bundles = [load_raw_bundle(Path(model_path)) for model_path in args.model]
    val_labels = [row_label(row) for row in val_rows]
    test_labels = [row_label(row) for row in test_rows]
    val_prob_arrays: list[np.ndarray] = []
    test_prob_arrays: list[np.ndarray] = []
    for bundle in bundles:
        config = bundle_config(bundle)
        val_prob_arrays.append(np.array(predict_dataset_probabilities(bundle, PathDecisionDataset(val_rows, config))))
        test_prob_arrays.append(np.array(predict_dataset_probabilities(bundle, PathDecisionDataset(test_rows, config))))
    blend_weights = select_ensemble_blend_weights(
        val_prob_arrays,
        val_labels,
        mode=args.blend_mode,
        step=args.blend_step,
    )
    val_probs = blend_probability_arrays(val_prob_arrays, blend_weights)
    threshold_settings = make_threshold_settings(
        args.threshold_mode,
        val_probs.tolist(),
        val_rows,
        val_labels,
        min_rows=args.threshold_min_rows,
    )
    test_probs = blend_probability_arrays(test_prob_arrays, blend_weights)
    metrics = evaluate_probabilities_with_threshold_settings(
        test_probs.tolist(),
        test_rows,
        test_labels,
        threshold_settings,
    )
    metrics["threshold"] = float(threshold_settings.get("fallback_threshold", threshold_settings.get("threshold", 0.5)))
    metrics["threshold_mode"] = args.threshold_mode
    metrics["threshold_settings"] = threshold_settings
    metrics["model_type"] = "model_blend"
    metrics["model_paths"] = list(args.model)
    metrics["blend_mode"] = args.blend_mode
    metrics["blend_step"] = args.blend_step
    metrics["blend_weights"] = blend_weights
    metrics["split_report"] = split_report(
        {"train": train_rows, "validation": val_rows, "test": test_rows},
        args.split_strategy,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train_tabular_ensemble_model(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    records = read_records(Path(args.input))
    if args.max_rows:
        records = records[: args.max_rows]
    if len(records) < 20:
        raise ValueError("Need at least 20 labeled decision rows for training.")

    train_rows, val_rows, test_rows = split_records(
        records,
        args.test_size,
        args.val_size,
        args.seed,
        strategy=args.split_strategy,
    )
    config = FeatureConfig()
    train_data = PathDecisionDataset(train_rows, config, fit=True)
    val_data = PathDecisionDataset(val_rows, config)
    test_data = PathDecisionDataset(test_rows, config)
    x_train, y_train = as_tabular_matrix(train_data)
    x_val, y_val = as_tabular_matrix(val_data)
    x_test, y_test = as_tabular_matrix(test_data)

    models = []
    model_configs = parse_hgb_ensemble_configs(args.configs)
    sample_weight = sample_weights_for_rows(train_rows, args.sample_weight)
    for idx, model_config in enumerate(model_configs):
        model = HistGradientBoostingClassifier(
            learning_rate=float(model_config["learning_rate"]),
            max_iter=int(model_config["max_iter"]),
            max_leaf_nodes=int(model_config["max_leaf_nodes"]),
            l2_regularization=float(model_config["l2_regularization"]),
            random_state=args.seed + idx,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        models.append(model)

    val_model_probs = [model.predict_proba(x_val)[:, 1] for model in models]
    blend_weights = select_ensemble_blend_weights(
        val_model_probs,
        y_val.tolist(),
        mode=args.blend_mode,
    )
    val_probs = blend_probability_arrays(val_model_probs, blend_weights)
    threshold_settings = make_threshold_settings(
        args.threshold_mode,
        val_probs.tolist(),
        val_rows,
        y_val.tolist(),
        min_rows=args.threshold_min_rows,
    )
    threshold = float(threshold_settings.get("fallback_threshold", threshold_settings.get("threshold", 0.5)))
    test_model_probs = [model.predict_proba(x_test)[:, 1] for model in models]
    test_probs = blend_probability_arrays(test_model_probs, blend_weights)
    metrics = evaluate_probabilities_with_threshold_settings(
        test_probs.tolist(),
        test_rows,
        y_test.tolist(),
        threshold_settings,
    )
    metrics["threshold"] = threshold
    metrics["threshold_mode"] = args.threshold_mode
    metrics["threshold_settings"] = threshold_settings
    metrics["model_type"] = "hist_gradient_boosting_ensemble"
    metrics["ensemble_size"] = len(models)
    metrics["model_configs"] = model_configs
    metrics["blend_mode"] = args.blend_mode
    metrics["blend_weights"] = blend_weights
    metrics["sample_weight"] = args.sample_weight
    metrics["split_report"] = split_report(
        {"train": train_rows, "validation": val_rows, "test": test_rows},
        args.split_strategy,
    )

    bundle = {
        "model_type": "hist_gradient_boosting_ensemble",
        "models": models,
        "model_configs": model_configs,
        "blend_weights": blend_weights,
        "config": config_to_dict(config),
        "metrics": metrics,
        "threshold_settings": threshold_settings,
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as handle:
        pickle.dump(bundle, handle)
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
