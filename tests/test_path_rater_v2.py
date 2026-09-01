import unittest

from path_rater_v2 import (
    FeatureConfig,
    PathDecisionDataset,
    as_tabular_matrix,
    audit_decision_rows,
    best_threshold_from_probs,
    blend_probability_arrays,
    extract_state_from_current_run,
    featurize_record,
    make_synthetic_records,
    parse_hgb_ensemble_configs,
    prepare_chosen_suffix_rows,
    prepare_sts1_run_history,
    prepare_sts2_run_history,
    enumerate_candidate_paths_from_saved_map,
    rank_chosen_path,
    sample_weights_for_rows,
    select_ensemble_blend_weights,
    slice_threshold_settings,
    simplex_weight_grid,
    split_report,
    split_records,
    slice_value,
    summarize_choice_ranks,
    threshold_for_row,
    sts1_run_to_decisions,
    sts2_run_to_decisions,
    validate_no_leakage,
    worst_slices,
)


class PathRaterV2Tests(unittest.TestCase):
    def test_featurize_current_node_state(self):
        row = {
            "character": "IRONCLAD",
            "ascension": 12,
            "act": 1,
            "floor": 7,
            "current_hp": 20,
            "max_hp": 80,
            "current_gold": 260,
            "deck": ["Strike", "Bash"],
            "relics": ["Burning Blood"],
            "potions": [],
            "candidate_path": "M-E-R-$-BOSS",
            "victory": True,
        }
        item = featurize_record(row)
        self.assertEqual(item["numeric"]["remaining_elites"], 1)
        self.assertEqual(item["numeric"]["remaining_rest_sites"], 1)
        self.assertEqual(item["numeric"]["gold_high"], 1.0)
        self.assertEqual(item["numeric"]["hp_low"], 1.0)
        self.assertEqual(item["numeric"]["elite_before_rest"], 1.0)
        self.assertEqual(item["numeric"]["low_hp_elite_pressure"], 1.0)
        self.assertEqual(item["numeric"]["high_gold_shop_access"], 1.0)
        self.assertEqual(item["numeric"]["early_danger"], 3.0)
        self.assertEqual(item["numeric"]["danger_before_rest"], 3.0)
        self.assertEqual(item["numeric"]["last_rest_distance_to_boss"], 2.0)
        self.assertEqual(item["numeric"]["path_len_x_missing_hp"], 3.75)
        self.assertEqual(item["numeric"]["ironclad_act2"], 0.0)
        self.assertEqual(item["categorical"]["skill_bucket"], "mid")

    def test_featurize_route_risk_interactions(self):
        row = {
            "character": "THE_SILENT",
            "ascension": 20,
            "act": 3,
            "floor": 48,
            "current_hp": 16,
            "max_hp": 80,
            "current_gold": 80,
            "deck": ["Strike_G"] * 5,
            "relics": ["Ring of the Snake"],
            "potions": [],
            "candidate_path": ["M", "E", "?", "M", "E", "M", "?", "M", "E", "?", "M", "BOSS"],
            "victory": False,
        }
        item = featurize_record(row)
        self.assertEqual(item["numeric"]["low_hp_no_rest"], 1.0)
        self.assertEqual(item["numeric"]["low_hp_long_path"], 1.0)
        self.assertEqual(item["numeric"]["no_rest_long_path"], 1.0)
        self.assertEqual(item["numeric"]["late_act_long_path"], 1.0)
        self.assertEqual(item["numeric"]["late_act_low_hp"], 1.0)
        self.assertEqual(item["numeric"]["silent_act3"], 1.0)
        self.assertEqual(item["numeric"]["danger_after_last_rest"], 11.0)
        self.assertAlmostEqual(item["numeric"]["ascension_x_missing_hp"], 16.0)

    def test_featurize_zero_hp_is_low_hp(self):
        item = featurize_record(
            {
                "current_hp": 0,
                "max_hp": 80,
                "candidate_path": ["E", "BOSS"],
                "victory": False,
            }
        )
        self.assertEqual(item["numeric"]["hp_low"], 1.0)
        self.assertEqual(item["numeric"]["low_hp_elite_pressure"], 1.0)
        self.assertEqual(item["numeric"]["hp_mid"], 0.0)
        self.assertEqual(item["numeric"]["hp_high"], 0.0)

    def test_leakage_guard_flags_future_keys(self):
        row = {"future_card_rewards": ["Inflame"], "candidate_path": ["M"], "victory": 1}
        with self.assertRaises(ValueError):
            validate_no_leakage(row)

    def test_synthetic_records_are_labeled(self):
        rows = make_synthetic_records(50)
        self.assertEqual(len(rows), 50)
        self.assertTrue(all("victory" in row for row in rows))

    def test_tabular_matrix_has_one_row_per_record(self):
        rows = make_synthetic_records(20)
        data = PathDecisionDataset(rows, FeatureConfig(), fit=True)
        matrix, labels = as_tabular_matrix(data)
        self.assertEqual(matrix.shape[0], 20)
        self.assertEqual(labels.shape[0], 20)

    def test_survival_target_uses_floors_reached_label_only(self):
        item = featurize_record(
            {
                "candidate_path": ["M", "BOSS"],
                "victory": False,
                "floor": 8,
                "floors_reached": 25,
            }
        )
        self.assertAlmostEqual(item["survival_target"], 25 / 51)

    def test_prepare_chosen_suffix_rows_uses_current_floor_values(self):
        rows = prepare_chosen_suffix_rows(
            [
                {
                    "character": "IRONCLAD",
                    "ascension": 20,
                    "path_per_floor": ["M", "E", "R", "BOSS"],
                    "hp_per_floor": [80, 40, 20, 10],
                    "gold": [0, 99, 220, 230],
                    "max_hp": 80,
                    "victory": False,
                }
            ]
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["current_hp"], 40)
        self.assertEqual(rows[1]["current_gold"], 99)
        self.assertEqual(rows[1]["candidate_path"], ["E", "R", "BOSS"])

    def test_prepare_chosen_suffix_rows_stops_candidate_at_boss(self):
        rows = prepare_chosen_suffix_rows(
            [
                {
                    "character": "IRONCLAD",
                    "path_per_floor": ["M", "BOSS", "M", "BOSS"],
                    "hp_per_floor": [80, 70, 60, 50],
                    "gold": [0, 0, 0, 0],
                    "max_hp": 80,
                    "victory": True,
                }
            ]
        )
        self.assertEqual(rows[0]["candidate_path"], ["M", "BOSS"])
        self.assertEqual(rows[2]["candidate_path"], ["M", "BOSS"])

    def test_sts1_decisions_use_current_state_without_final_deck_leakage(self):
        rows = sts1_run_to_decisions(
            {
                "play_id": "run-1",
                "character_chosen": "DEFECT",
                "ascension_level": 6,
                "build_version": "2020-07-30",
                "path_per_floor": ["M", "M", "?", "B", None, "E"],
                "current_hp_per_floor": [68, 62, 60, 0],
                "max_hp_per_floor": [75, 75, 75, 75],
                "gold_per_floor": [109, 125, 125, 125],
                "card_choices": [{"picked": "Hologram", "floor": 1.0}],
                "relics_obtained": [{"key": "Ice Cream", "floor": 3.0}],
                "master_deck": ["FUTURE_CARD"],
                "relics": ["FUTURE_RELIC"],
                "floor_reached": 4,
                "victory": False,
            }
        )
        self.assertEqual(rows[0]["candidate_path"], ["M", "M", "?", "BOSS"])
        self.assertNotIn("E", rows[0]["candidate_path"])
        self.assertEqual(rows[1]["current_hp"], 62)
        self.assertEqual(rows[1]["current_gold"], 125)
        self.assertIn("Hologram", rows[1]["deck"])
        self.assertNotIn("FUTURE_CARD", rows[1]["deck"])
        self.assertNotIn("FUTURE_RELIC", rows[1]["relics"])
        self.assertIn("Cracked Core", rows[1]["relics"])

    def test_sts1_act_survival_label_uses_current_act_only(self):
        base = {
            "play_id": "run-1",
            "character_chosen": "IRONCLAD",
            "path_per_floor": ["M", "B", None, "M", "B"],
            "current_hp_per_floor": [70, 70, 70, 60, 60],
            "max_hp_per_floor": [80, 80, 80, 80, 80],
            "gold_per_floor": [10, 10, 10, 20, 20],
            "victory": False,
        }
        died_after_act_one = sts1_run_to_decisions(dict(base, floor_reached=3), label_mode="act-survival")
        died_at_act_one_boss = sts1_run_to_decisions(dict(base, floor_reached=2), label_mode="act-survival")
        self.assertTrue(died_after_act_one[0]["victory"])
        self.assertFalse(died_at_act_one_boss[0]["victory"])

    def test_sts1_prepare_excludes_daily_endless_and_trial_by_default(self):
        base = {
            "play_id": "clean",
            "character_chosen": "IRONCLAD",
            "path_per_floor": ["M", "B"],
            "current_hp_per_floor": [70, 0],
            "max_hp_per_floor": [80, 80],
            "gold_per_floor": [10, 10],
            "floor_reached": 2,
            "victory": False,
        }
        rows = prepare_sts1_run_history(
            [
                dict(base, play_id="clean"),
                dict(base, play_id="daily", is_daily=True),
                dict(base, play_id="endless", is_endless=True),
                dict(base, play_id="trial", is_trial=True),
            ]
        )
        self.assertEqual({row["run_id"] for row in rows}, {"clean"})

    def test_sts1_prepare_filters_invalid_state_by_default(self):
        base = {
            "character_chosen": "IRONCLAD",
            "path_per_floor": ["M", "B"],
            "current_hp_per_floor": [70, 70],
            "max_hp_per_floor": [80, 80],
            "gold_per_floor": [10, 10],
            "floor_reached": 2,
        }
        rows = prepare_sts1_run_history(
            [
                dict(base, play_id="clean"),
                dict(base, play_id="bad-gold", gold_per_floor=[13000, 13000]),
                dict(base, play_id="bad-hp", current_hp_per_floor=[90, 90]),
            ]
        )
        self.assertEqual({row["run_id"] for row in rows}, {"clean"})

        rows_with_invalid = prepare_sts1_run_history(
            [dict(base, play_id="bad-gold", gold_per_floor=[13000, 13000])],
            include_invalid_state=True,
        )
        self.assertEqual(len(rows_with_invalid), 2)

    def test_sts1_candidate_path_stops_at_next_boss(self):
        rows = sts1_run_to_decisions(
            {
                "play_id": "run-acts",
                "character_chosen": "IRONCLAD",
                "path_per_floor": ["M", "B", None, "E", "B"],
                "current_hp_per_floor": [80, 70, 70, 40, 0],
                "max_hp_per_floor": [80, 80, 80, 80, 80],
                "gold_per_floor": [99, 120, 120, 150, 150],
                "floor_reached": 5,
            }
        )
        self.assertEqual(rows[0]["candidate_path"], ["M", "BOSS"])
        self.assertEqual(rows[2]["candidate_path"], ["E", "BOSS"])

    def test_grouped_split_keeps_runs_apart(self):
        records = []
        for run_idx in range(12):
            for row_idx in range(3):
                records.append(
                    {
                        "run_id": f"run-{run_idx}",
                        "candidate_path": ["M", "BOSS"],
                        "victory": run_idx % 2 == 0,
                        "floor": row_idx,
                    }
                )
        train_rows, val_rows, test_rows = split_records(records, 0.25, 0.25, 7)
        train_ids = {row["run_id"] for row in train_rows}
        val_ids = {row["run_id"] for row in val_rows}
        test_ids = {row["run_id"] for row in test_rows}
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_grouped_split_stratifies_labels_when_possible(self):
        records = []
        for run_idx in range(20):
            for row_idx in range(2):
                records.append(
                    {
                        "run_id": f"run-{run_idx}",
                        "candidate_path": ["M", "BOSS"],
                        "victory": run_idx % 2 == 0,
                        "floor": row_idx,
                    }
                )
        _train_rows, _val_rows, test_rows = split_records(records, 0.25, 0.25, 7)
        labels = {bool(row["victory"]) for row in test_rows}
        self.assertEqual(labels, {False, True})

    def test_chronological_split_uses_last_runs_for_test(self):
        records = []
        for run_idx in range(10):
            records.append(
                {
                    "run_id": str(run_idx),
                    "candidate_path": ["M", "BOSS"],
                    "victory": run_idx % 2 == 0,
                }
            )
        train_rows, val_rows, test_rows = split_records(records, 0.2, 0.25, 7, strategy="chronological")
        self.assertEqual({row["run_id"] for row in test_rows}, {"8", "9"})
        self.assertEqual({row["run_id"] for row in val_rows}, {"6", "7"})
        self.assertEqual({row["run_id"] for row in train_rows}, {"0", "1", "2", "3", "4", "5"})

    def test_split_report_counts_rows_runs_and_group_overlap(self):
        report = split_report(
            {
                "train": [
                    {"run_id": "a", "victory": True},
                    {"run_id": "a", "victory": False},
                ],
                "validation": [{"run_id": "b", "victory": False}],
                "test": [{"run_id": "c", "victory": True}],
            },
            "stratified-group",
        )
        self.assertEqual(report["splits"]["train"]["rows"], 2)
        self.assertEqual(report["splits"]["train"]["runs"], 1)
        self.assertEqual(report["splits"]["train"]["positive_rows"], 1)
        self.assertEqual(report["run_group_overlaps"]["train_validation"], 0)
        self.assertEqual(report["run_group_overlaps"]["train_test"], 0)

    def test_parse_hgb_ensemble_configs(self):
        configs = parse_hgb_ensemble_configs("0.03:800:63:0,0.08:500:31:0.01")
        self.assertEqual(len(configs), 2)
        self.assertEqual(configs[0]["max_iter"], 800)
        self.assertAlmostEqual(configs[1]["learning_rate"], 0.08)
        self.assertAlmostEqual(configs[1]["l2_regularization"], 0.01)

    def test_simplex_weight_grid_sums_to_one(self):
        weights = simplex_weight_grid(3, step=0.5)
        self.assertIn([0.0, 0.0, 1.0], weights)
        self.assertIn([0.5, 0.5, 0.0], weights)
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-9 for row in weights))

    def test_blend_probability_arrays_uses_weights(self):
        blended = blend_probability_arrays(
            [
                __import__("numpy").array([0.0, 1.0]),
                __import__("numpy").array([1.0, 0.0]),
            ],
            [0.25, 0.75],
        )
        self.assertAlmostEqual(float(blended[0]), 0.75)
        self.assertAlmostEqual(float(blended[1]), 0.25)

    def test_select_ensemble_blend_weights_can_choose_single_model(self):
        np = __import__("numpy")
        weights = select_ensemble_blend_weights(
            [
                np.array([0.9, 0.8, 0.2, 0.1]),
                np.array([0.1, 0.2, 0.8, 0.9]),
            ],
            [1, 1, 0, 0],
            mode="validation",
            step=0.5,
        )
        self.assertEqual(weights, [1.0, 0.0])

    def test_best_threshold_uses_probability_values(self):
        threshold = best_threshold_from_probs([0.101, 0.102, 0.901], [1, 0, 1])
        self.assertEqual(threshold, 0.101)

    def test_slice_threshold_settings_uses_bucket_and_fallback(self):
        rows = [
            {"act": 1, "candidate_path": ["M"], "current_hp": 80, "max_hp": 80},
            {"act": 1, "candidate_path": ["M"], "current_hp": 80, "max_hp": 80},
            {"act": 2, "candidate_path": ["M"], "current_hp": 80, "max_hp": 80},
            {"act": 2, "candidate_path": ["M"], "current_hp": 80, "max_hp": 80},
        ]
        settings = slice_threshold_settings([0.2, 0.8, 0.3, 0.4], rows, [1, 0, 0, 1], min_rows=2)
        self.assertEqual(settings["mode"], "slice")
        self.assertEqual(settings["fallback_threshold"], 0.2)
        self.assertEqual(threshold_for_row(rows[0], settings), 0.2)
        self.assertEqual(threshold_for_row(rows[2], settings), 0.4)
        self.assertEqual(
            threshold_for_row({"act": 3, "candidate_path": ["M"], "current_hp": 80, "max_hp": 80}, settings),
            0.2,
        )

    def test_inverse_run_sample_weights_give_each_run_equal_weight(self):
        weights = sample_weights_for_rows(
            [
                {"run_id": "a"},
                {"run_id": "a"},
                {"run_id": "b"},
            ],
            "inverse-run",
        )
        self.assertEqual(weights.tolist(), [0.5, 0.5, 1.0])

    def test_balanced_sample_weights_give_classes_equal_total_weight(self):
        rows = [
            {"run_id": "a", "victory": True},
            {"run_id": "b", "victory": True},
            {"run_id": "c", "victory": False},
        ]
        weights = sample_weights_for_rows(rows, "balanced")
        positive_weight = weights[0] + weights[1]
        negative_weight = weights[2]
        self.assertAlmostEqual(float(positive_weight), float(negative_weight))

    def test_balanced_inverse_run_weights_balance_runs_and_classes(self):
        rows = [
            {"run_id": "a", "victory": True},
            {"run_id": "a", "victory": True},
            {"run_id": "b", "victory": False},
        ]
        weights = sample_weights_for_rows(rows, "balanced-inverse-run")
        self.assertAlmostEqual(float(weights[0] + weights[1]), float(weights[2]))

    def test_slice_value_buckets_state_fields(self):
        record = {
            "character": "DEFECT",
            "act": 2,
            "ascension": 20,
            "floor": 23,
            "current_hp": 20,
            "max_hp": 80,
            "current_gold": 260,
            "candidate_path": ["M", "E", "R", "$", "BOSS"],
            "skill_bucket": "high",
        }
        self.assertEqual(slice_value(record, "character"), "DEFECT")
        self.assertEqual(slice_value(record, "act"), "act_2")
        self.assertEqual(slice_value(record, "ascension_bucket"), "a20")
        self.assertEqual(slice_value(record, "floor_band"), "floor_17_33")
        self.assertEqual(slice_value(record, "hp_bucket"), "hp_low")
        self.assertEqual(slice_value(record, "gold_bucket"), "gold_high")
        self.assertEqual(slice_value(record, "path_len_bucket"), "path_1_5")
        self.assertEqual(slice_value(record, "skill_bucket"), "high")

    def test_worst_slices_sorts_by_positive_f1(self):
        result = worst_slices(
            {
                "act": {
                    "act_1": {"positive_f1": 0.91, "negative_f1": 0.7, "macro_f1": 0.8, "n": 1000},
                    "act_3": {"positive_f1": 0.82, "negative_f1": 0.6, "macro_f1": 0.71, "n": 800},
                },
                "hp_bucket": {
                    "hp_low": {"positive_f1": 0.84, "negative_f1": 0.85, "macro_f1": 0.845, "n": 900}
                },
            },
            limit=2,
        )
        self.assertEqual([row["value"] for row in result], ["act_3", "hp_low"])
        self.assertEqual(result[0]["field"], "act")
        self.assertEqual(result[0]["rows"], 800)

    def test_rank_chosen_path_reports_exact_candidate_rank(self):
        scored = [
            {"candidate": ["M", "E", "BOSS"], "win_probability": 0.8},
            {"candidate": ["M", "R", "BOSS"], "win_probability": 0.6},
            {"candidate": ["?", "$", "BOSS"], "win_probability": 0.2},
        ]
        result = rank_chosen_path(scored, ["M", "R", "BOSS"])
        self.assertTrue(result["chosen_found"])
        self.assertEqual(result["chosen_rank"], 2)
        self.assertEqual(result["candidate_count"], 3)
        self.assertAlmostEqual(result["probability_gap_to_top"], 0.2)

    def test_rank_chosen_path_reports_missing_candidate(self):
        scored = [{"candidate": ["M", "E", "BOSS"], "win_probability": 0.8}]
        result = rank_chosen_path(scored, ["M", "$", "BOSS"])
        self.assertFalse(result["chosen_found"])
        self.assertIsNone(result["chosen_rank"])

    def test_summarize_choice_ranks_reports_top_rates(self):
        summary = summarize_choice_ranks(
            [
                {
                    "chosen_found": True,
                    "chosen_rank": 1,
                    "rank_percentile": 1.0,
                    "probability_gap_to_top": 0.0,
                },
                {
                    "chosen_found": True,
                    "chosen_rank": 3,
                    "rank_percentile": 0.5,
                    "probability_gap_to_top": 0.2,
                },
                {"chosen_found": False},
            ]
        )
        self.assertEqual(summary["examples"], 3)
        self.assertEqual(summary["chosen_found"], 2)
        self.assertEqual(summary["chosen_missing"], 1)
        self.assertAlmostEqual(summary["top1_rate"], 0.5)
        self.assertAlmostEqual(summary["top3_rate"], 1.0)
        self.assertAlmostEqual(summary["mean_reciprocal_rank"], (1.0 + 1.0 / 3.0) / 2.0)
        self.assertAlmostEqual(summary["mean_probability_gap_to_top"], 0.1)

    def test_sts2_decisions_do_not_use_future_act_or_future_cards(self):
        record = {
            "ascension": 3,
            "build_id": "v0.test",
            "game_mode": "standard",
            "start_time": 1,
            "win": True,
            "map_point_history": [
                [
                    {
                        "map_point_type": "monster",
                        "player_stats": [{"current_hp": 70, "max_hp": 80, "current_gold": 30}],
                    },
                    {
                        "map_point_type": "elite",
                        "player_stats": [{"current_hp": 50, "max_hp": 80, "current_gold": 90}],
                    },
                ],
                [
                    {
                        "map_point_type": "shop",
                        "player_stats": [{"current_hp": 50, "max_hp": 80, "current_gold": 90}],
                    }
                ],
            ],
            "players": [
                {
                    "character": "CHARACTER.IRONCLAD",
                    "deck": [
                        {"floor_added_to_deck": 1, "id": "CARD.STRIKE"},
                        {"floor_added_to_deck": 3, "id": "CARD.FUTURE"},
                    ],
                    "relics": [{"floor_added_to_deck": 1, "id": "RELIC.START"}],
                    "potions": [],
                }
            ],
        }
        rows = sts2_run_to_decisions(record)
        self.assertEqual(rows[0]["candidate_path"], ["M", "E"])
        self.assertEqual(rows[1]["candidate_path"], ["E"])
        self.assertEqual(rows[0]["floors_reached"], 3)
        self.assertNotIn("CARD.FUTURE", rows[1]["deck"])

    def test_sts2_prior_skill_uses_earlier_runs(self):
        base = {
            "ascension": 0,
            "build_id": "v0.test",
            "game_mode": "standard",
            "map_point_history": [[{"map_point_type": "monster", "player_stats": [{"max_hp": 80}]}]],
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []}],
        }
        first = dict(base, start_time=1, win=True)
        second = dict(base, start_time=2, win=False)
        rows = prepare_sts2_run_history([second, first])
        self.assertEqual(rows[0]["run_id"], "1")
        self.assertEqual(rows[0]["player_prior_runs"], 0.0)
        self.assertEqual(rows[1]["run_id"], "2")
        self.assertEqual(rows[1]["player_prior_runs"], 1.0)
        self.assertEqual(rows[1]["player_prior_win_rate"], 1.0)

    def test_sts2_public_rows_without_player_id_do_not_get_skill_proxy(self):
        base = {
            "ascension": 0,
            "build_id": "v0.test",
            "game_mode": "standard",
            "map_point_history": [[{"map_point_type": "monster", "player_stats": [{"max_hp": 80}]}]],
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []}],
        }
        first = dict(base, start_time=1, win=True, _serverId=101)
        second = dict(base, start_time=2, win=False, _serverId=102)
        rows = prepare_sts2_run_history([first, second])
        self.assertEqual(rows[0]["player_prior_runs"], 0.0)
        self.assertEqual(rows[1]["player_prior_runs"], 0.0)
        self.assertEqual(rows[1]["player_prior_win_rate"], 0.0)

    def test_sts2_prepare_excludes_abandoned_by_default(self):
        base = {
            "ascension": 0,
            "build_id": "v0.test",
            "game_mode": "standard",
            "map_point_history": [[{"map_point_type": "monster", "player_stats": [{"max_hp": 80}]}]],
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []}],
        }
        abandoned = dict(base, start_time=1, win=False, was_abandoned=True)
        normal = dict(base, start_time=2, win=True, was_abandoned=False)
        rows = prepare_sts2_run_history([abandoned, normal])
        self.assertEqual([row["run_id"] for row in rows], ["2"])
        included = prepare_sts2_run_history([abandoned, normal], include_abandoned=True)
        self.assertEqual([row["run_id"] for row in included], ["1", "2"])

    def test_sts2_prepare_filters_build_id(self):
        base = {
            "ascension": 0,
            "game_mode": "standard",
            "map_point_history": [[{"map_point_type": "monster", "player_stats": [{"max_hp": 80}]}]],
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []}],
        }
        old = dict(base, start_time=1, win=True, build_id="old")
        new = dict(base, start_time=2, win=True, build_id="new")
        rows = prepare_sts2_run_history([old, new], build_ids={"new"})
        self.assertEqual([row["run_id"] for row in rows], ["2"])

    def test_sts2_prepare_filters_game_mode(self):
        base = {
            "ascension": 0,
            "build_id": "v0.test",
            "map_point_history": [[{"map_point_type": "monster", "player_stats": [{"max_hp": 80}]}]],
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []}],
            "win": True,
        }
        standard = dict(base, start_time=1, game_mode="standard")
        daily = dict(base, start_time=2, game_mode="daily")
        rows = prepare_sts2_run_history([standard, daily], game_modes={"standard"})
        self.assertEqual([row["run_id"] for row in rows], ["1"])

    def test_sts2_prepare_excludes_modded_and_coop_by_default(self):
        base = {
            "ascension": 0,
            "build_id": "v0.test",
            "game_mode": "standard",
            "map_point_history": [[{"map_point_type": "monster", "player_stats": [{"max_hp": 80}]}]],
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []}],
            "win": True,
        }
        vanilla = dict(base, start_time=1, _run_file="profile1/saves/history/1.run")
        modded = dict(base, start_time=2, _run_file="modded/profile1/saves/history/2.run")
        coop = dict(
            base,
            start_time=3,
            game_mode="coop",
            players=[
                {"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []},
                {"character": "CHARACTER.SILENT", "deck": [], "relics": [], "potions": []},
            ],
        )
        rows = prepare_sts2_run_history([vanilla, modded, coop])
        self.assertEqual([row["run_id"] for row in rows], ["1"])

    def test_sts2_prepare_excludes_cheated_by_default(self):
        base = {
            "ascension": 0,
            "build_id": "v0.test",
            "game_mode": "standard",
            "map_point_history": [[{"map_point_type": "monster", "player_stats": [{"max_hp": 80}]}]],
            "players": [{"character": "CHARACTER.IRONCLAD", "deck": [], "relics": [], "potions": []}],
            "win": True,
        }
        clean = dict(base, start_time=1, _isCheated=False)
        cheated = dict(base, start_time=2, _isCheated=True)
        rows = prepare_sts2_run_history([clean, cheated])
        self.assertEqual([row["run_id"] for row in rows], ["1"])
        included = prepare_sts2_run_history([clean, cheated], include_cheated=True)
        self.assertEqual([row["run_id"] for row in included], ["1", "2"])

    def test_enumerate_saved_map_from_current_node(self):
        saved_map = {
            "saved_map": {
                "nodes": [
                    {"id": "a", "map_point_type": "monster", "children": ["b", "c"]},
                    {"id": "b", "map_point_type": "elite", "children": ["d"]},
                    {"id": "c", "map_point_type": "shop", "children": ["d"]},
                    {"id": "d", "map_point_type": "boss", "children": []},
                ]
            }
        }
        paths = enumerate_candidate_paths_from_saved_map(saved_map, current_node="a")
        self.assertEqual(sorted(paths), sorted([["M", "E", "BOSS"], ["M", "$", "BOSS"]]))

    def test_audit_marks_vanilla_solo_population(self):
        rows = [
            {
                "run_id": "1",
                "profile_type": "vanilla",
                "player_count": 1,
                "game_mode": "standard",
                "build_id": "v0.test",
                "character": "CHARACTER.IRONCLAD",
                "candidate_path": ["M"],
                "victory": True,
            }
        ]
        audit = audit_decision_rows(rows)
        self.assertTrue(audit["valid_main_population"])
        self.assertEqual(audit["rows"], 1)

    def test_extract_state_from_current_run(self):
        state = extract_state_from_current_run(
            {
                "build_id": "v0.test",
                "game_mode": "standard",
                "current_act_index": 0,
                "ascension": 4,
                "map_point_history": [
                    [
                        {
                            "map_point_type": "monster",
                            "player_stats": [{"current_hp": 30, "max_hp": 80, "current_gold": 120}],
                        }
                    ]
                ],
                "players": [
                    {
                        "character": "CHARACTER.IRONCLAD",
                        "deck": [
                            {"floor_added_to_deck": 1, "id": "CARD.STRIKE"},
                            {"floor_added_to_deck": 3, "id": "CARD.FUTURE"},
                        ],
                        "relics": [{"floor_added_to_deck": 1, "id": "RELIC.START"}],
                        "potions": [{"id": "POTION.TEST"}],
                    }
                ],
            }
        )
        self.assertEqual(state["current_hp"], 30)
        self.assertEqual(state["current_gold"], 120)
        self.assertEqual(state["act"], 1)
        self.assertNotIn("CARD.FUTURE", state["deck"])


if __name__ == "__main__":
    unittest.main()
