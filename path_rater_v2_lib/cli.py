# Split from path_rater_v2.py. Keep public behavior compatible with the root wrapper.

from .common import *
from .data_io import read_records, write_ndjson
from .preparation import *
from .trainers import *
from .maps import *
from .metrics import analyze_prediction_slices
from .synthetic import self_test

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use the v2 path rater.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--model-out", default="models/path_rater_v2.pkl")
    train.add_argument("--report-out", default="reports/path_rater_v2_metrics.json")
    train.add_argument("--max-rows", type=int)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--val-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--d-model", type=int, default=64)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--aux-floor-weight", type=float, default=0.2)
    train.add_argument("--pos-weight", default="auto", help="Use 'auto', a numeric positive-class weight, or 0.")
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--no-position", action="store_true", help="Disable learned path position embeddings.")
    train.add_argument("--no-attention-pooling", action="store_true", help="Disable learned path attention pooling.")
    train.add_argument("--split-strategy", choices=["stratified-group", "chronological"], default="stratified-group")

    tabular = sub.add_parser("train-tabular")
    tabular.add_argument("--input", required=True)
    tabular.add_argument("--model-out", default="models/path_rater_v2_hgb.pkl")
    tabular.add_argument("--report-out", default="reports/path_rater_v2_hgb_metrics.json")
    tabular.add_argument("--max-rows", type=int)
    tabular.add_argument("--test-size", type=float, default=0.2)
    tabular.add_argument("--val-size", type=float, default=0.2)
    tabular.add_argument("--seed", type=int, default=7)
    tabular.add_argument("--learning-rate", type=float, default=0.05)
    tabular.add_argument("--max-iter", type=int, default=300)
    tabular.add_argument("--max-leaf-nodes", type=int, default=31)
    tabular.add_argument("--l2-regularization", type=float, default=0.0)
    tabular.add_argument(
        "--sample-weight",
        choices=["none", "inverse-run", "balanced", "balanced-inverse-run"],
        default="none",
    )
    tabular.add_argument("--threshold-mode", choices=["global", "slice"], default="global")
    tabular.add_argument("--threshold-min-rows", type=int, default=500)
    tabular.add_argument("--split-strategy", choices=["stratified-group", "chronological"], default="stratified-group")

    extra_trees = sub.add_parser("train-extra-trees")
    extra_trees.add_argument("--input", required=True)
    extra_trees.add_argument("--model-out", default="models/path_rater_v2_extra_trees.pkl")
    extra_trees.add_argument("--report-out", default="reports/path_rater_v2_extra_trees_metrics.json")
    extra_trees.add_argument("--max-rows", type=int)
    extra_trees.add_argument("--test-size", type=float, default=0.2)
    extra_trees.add_argument("--val-size", type=float, default=0.2)
    extra_trees.add_argument("--seed", type=int, default=7)
    extra_trees.add_argument("--n-estimators", type=int, default=300)
    extra_trees.add_argument("--max-depth", type=int, default=32)
    extra_trees.add_argument("--min-samples-leaf", type=int, default=5)
    extra_trees.add_argument("--max-features", default="sqrt")
    extra_trees.add_argument("--class-weight", choices=["balanced", "balanced_subsample"], default=None)
    extra_trees.add_argument("--n-jobs", type=int, default=-1)
    extra_trees.add_argument(
        "--sample-weight",
        choices=["none", "inverse-run", "balanced", "balanced-inverse-run"],
        default="none",
    )
    extra_trees.add_argument("--threshold-mode", choices=["global", "slice"], default="global")
    extra_trees.add_argument("--threshold-min-rows", type=int, default=500)
    extra_trees.add_argument("--split-strategy", choices=["stratified-group", "chronological"], default="stratified-group")

    tabular_ensemble = sub.add_parser("train-tabular-ensemble")
    tabular_ensemble.add_argument("--input", required=True)
    tabular_ensemble.add_argument("--model-out", default="models/path_rater_v2_hgb_ensemble.pkl")
    tabular_ensemble.add_argument("--report-out", default="reports/path_rater_v2_hgb_ensemble_metrics.json")
    tabular_ensemble.add_argument("--max-rows", type=int)
    tabular_ensemble.add_argument("--test-size", type=float, default=0.2)
    tabular_ensemble.add_argument("--val-size", type=float, default=0.2)
    tabular_ensemble.add_argument("--seed", type=int, default=7)
    tabular_ensemble.add_argument(
        "--configs",
        help="Comma-separated HGB configs as learning_rate:max_iter:max_leaf_nodes:l2_regularization.",
    )
    tabular_ensemble.add_argument("--blend-mode", choices=["equal", "validation"], default="equal")
    tabular_ensemble.add_argument(
        "--sample-weight",
        choices=["none", "inverse-run", "balanced", "balanced-inverse-run"],
        default="none",
    )
    tabular_ensemble.add_argument("--threshold-mode", choices=["global", "slice"], default="global")
    tabular_ensemble.add_argument("--threshold-min-rows", type=int, default=500)
    tabular_ensemble.add_argument(
        "--split-strategy",
        choices=["stratified-group", "chronological"],
        default="stratified-group",
    )

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--mode",
        choices=["chosen-suffix", "sts1-run-history", "sts2-run-history"],
        default="chosen-suffix",
    )
    prepare.add_argument("--max-floor", type=int)
    prepare.add_argument("--max-rows", type=int)
    prepare.add_argument("--max-files", type=int)
    prepare.add_argument("--label-mode", choices=["victory", "act-survival"], default="victory")
    prepare.add_argument("--include-abandoned", action="store_true")
    prepare.add_argument("--include-cheated", action="store_true")
    prepare.add_argument("--include-daily", action="store_true")
    prepare.add_argument("--include-endless", action="store_true")
    prepare.add_argument("--include-trial", action="store_true")
    prepare.add_argument("--build-id", action="append", dest="build_ids")
    prepare.add_argument("--character", action="append", dest="characters")
    prepare.add_argument("--game-mode", action="append", dest="game_modes")
    prepare.add_argument("--include-modded", action="store_true")
    prepare.add_argument("--include-coop", action="store_true")
    prepare.add_argument("--include-invalid-state", action="store_true")

    score = sub.add_parser("score")
    score.add_argument("--model", required=True)
    score.add_argument("--state", required=True)
    score.add_argument("--candidates", required=True)

    enum_map = sub.add_parser("enumerate-map")
    enum_map.add_argument("--input", required=True)
    enum_map.add_argument("--output", required=True)
    enum_map.add_argument("--current-node")
    enum_map.add_argument("--max-paths", type=int, default=5000)

    score_map = sub.add_parser("score-map")
    score_map.add_argument("--model", required=True)
    score_map.add_argument("--state", required=True)
    score_map.add_argument("--map", required=True)
    score_map.add_argument("--current-node")
    score_map.add_argument("--max-paths", type=int, default=5000)

    eval_map = sub.add_parser("eval-map-choice")
    eval_map.add_argument("--model", required=True)
    eval_map.add_argument("--state", required=True)
    eval_map.add_argument("--map", required=True)
    eval_map.add_argument("--chosen-path", required=True)
    eval_map.add_argument("--current-node")
    eval_map.add_argument("--max-paths", type=int, default=5000)

    eval_maps = sub.add_parser("eval-map-choices")
    eval_maps.add_argument("--model", required=True)
    eval_maps.add_argument("--input", required=True)
    eval_maps.add_argument("--output")
    eval_maps.add_argument("--max-paths", type=int, default=5000)
    eval_maps.add_argument("--include-scored-candidates", action="store_true")

    blend = sub.add_parser("eval-blend")
    blend.add_argument("--input", required=True)
    blend.add_argument("--model", action="append", required=True)
    blend.add_argument("--output")
    blend.add_argument("--max-rows", type=int)
    blend.add_argument("--test-size", type=float, default=0.2)
    blend.add_argument("--val-size", type=float, default=0.2)
    blend.add_argument("--seed", type=int, default=7)
    blend.add_argument("--blend-mode", choices=["equal", "validation"], default="validation")
    blend.add_argument("--blend-step", type=float, default=0.05)
    blend.add_argument("--threshold-mode", choices=["global", "slice"], default="global")
    blend.add_argument("--threshold-min-rows", type=int, default=500)
    blend.add_argument("--split-strategy", choices=["stratified-group", "chronological"], default="stratified-group")

    analyze = sub.add_parser("analyze-slices")
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output")
    analyze.add_argument("--test-size", type=float, default=0.2)
    analyze.add_argument("--val-size", type=float, default=0.2)
    analyze.add_argument("--seed", type=int, default=7)
    analyze.add_argument("--min-rows", type=int, default=200)
    analyze.add_argument(
        "--split-strategy",
        choices=["stratified-group", "chronological"],
        default="stratified-group",
    )

    state = sub.add_parser("state-from-save")
    state.add_argument("--input", required=True)
    state.add_argument("--output", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output")

    test = sub.add_parser("self-test")
    test.add_argument("--output-dir", default="tmp/path_rater_v2")
    test.add_argument("--rows", type=int, default=1000)
    test.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        metrics = train_model(args)
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "train-tabular":
        metrics = train_tabular_model(args)
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "train-extra-trees":
        metrics = train_extra_trees_model(args)
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "train-tabular-ensemble":
        metrics = train_tabular_ensemble_model(args)
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "score":
        result = score_candidates(Path(args.model), Path(args.state), Path(args.candidates))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "enumerate-map":
        paths = enumerate_saved_map_file(
            Path(args.input),
            current_node=args.current_node,
            max_paths=args.max_paths,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(paths, indent=2), encoding="utf-8")
        print(json.dumps({"paths": len(paths), "output": args.output}, indent=2))
        return 0
    if args.command == "score-map":
        result = score_saved_map(
            Path(args.model),
            Path(args.state),
            Path(args.map),
            current_node=args.current_node,
            max_paths=args.max_paths,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "eval-map-choice":
        chosen_path = load_candidate_path_argument(args.chosen_path)
        result = evaluate_chosen_path_on_map(
            Path(args.model),
            Path(args.state),
            Path(args.map),
            chosen_path=chosen_path,
            current_node=args.current_node,
            max_paths=args.max_paths,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "eval-map-choices":
        input_path = Path(args.input)
        records = read_records(input_path)
        result = evaluate_map_choice_records(
            Path(args.model),
            records,
            base_dir=input_path.parent,
            max_paths=args.max_paths,
            include_scored_candidates=args.include_scored_candidates,
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "eval-blend":
        result = evaluate_model_blend(args)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "state-from-save":
        state = extract_state_file(Path(args.input))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps({"output": args.output}, indent=2))
        return 0
    if args.command == "audit":
        records = read_records(Path(args.input))
        result = audit_decision_rows(records)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "analyze-slices":
        result = analyze_prediction_slices(
            Path(args.model),
            Path(args.input),
            split_strategy=args.split_strategy,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
            min_rows=args.min_rows,
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "prepare":
        if args.mode == "chosen-suffix":
            records = read_records(Path(args.input))
            prepared = prepare_chosen_suffix_rows(records, max_floor=args.max_floor)
        elif args.mode == "sts1-run-history":
            records = read_sts1_runs(Path(args.input), max_files=args.max_files)
            prepared = prepare_sts1_run_history(
                records,
                max_rows=args.max_rows,
                max_floor=args.max_floor,
                label_mode=args.label_mode,
                include_daily=args.include_daily,
                include_endless=args.include_endless,
                include_trial=args.include_trial,
                characters=set(args.characters or []),
                include_invalid_state=args.include_invalid_state,
            )
        elif args.mode == "sts2-run-history":
            records = read_sts2_runs(Path(args.input))
            prepared = prepare_sts2_run_history(
                records,
                max_rows=args.max_rows,
                include_abandoned=args.include_abandoned,
                build_ids=set(args.build_ids or []),
                characters=set(args.characters or []),
                game_modes=set(args.game_modes or []),
                vanilla_only=not args.include_modded,
                include_coop=args.include_coop,
                include_cheated=args.include_cheated,
            )
        else:
            raise ValueError(args.mode)
        write_ndjson(Path(args.output), prepared)
        print(json.dumps({"rows": len(prepared), "output": args.output}, indent=2))
        return 0
    if args.command == "self-test":
        metrics = self_test(args)
        print(json.dumps(metrics, indent=2))
        return 0
    raise ValueError(args.command)
