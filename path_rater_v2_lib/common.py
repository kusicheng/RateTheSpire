"""Second-iteration path rating model for Slay the Spire style maps.

This module trains and serves a path rater from decision-time data only.
It intentionally accepts a simple decision-row format so it can work with
STS1 oracle output or STS2 run exports after a parser maps them to rows.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOM_TYPES = ["PAD", "M", "?", "E", "R", "$", "T", "A", "BOSS", "UNKNOWN"]
ROOM_TO_ID = {room: idx for idx, room in enumerate(ROOM_TYPES)}
MAX_PATH_LEN = 20
HASH_BUCKETS = 64
MAX_EXPECTED_FLOOR = 51.0

NUMERIC_FIELDS = [
    "ascension",
    "act",
    "floor",
    "current_hp",
    "max_hp",
    "current_gold",
    "deck_size",
    "relic_count",
    "potion_count",
    "path_len",
    "floor_in_act",
    "act_progress",
    "path_len_ratio",
    "remaining_elites",
    "remaining_rest_sites",
    "remaining_shops",
    "remaining_monsters",
    "remaining_events",
    "elite_density",
    "monster_density",
    "rest_density",
    "shop_density",
    "next_is_elite",
    "next_is_rest",
    "next_is_shop",
    "early_elites",
    "early_rest_sites",
    "early_shops",
    "first_three_elites",
    "first_three_rest_sites",
    "first_three_shops",
    "first_three_monsters",
    "first_three_events",
    "distance_to_elite",
    "distance_to_rest",
    "distance_to_shop",
    "elite_before_rest",
    "rest_after_elite",
    "rest_before_elite",
    "shop_before_elite",
    "elites_before_rest",
    "monsters_before_rest",
    "events_before_rest",
    "low_hp_elite_pressure",
    "low_hp_rest_relief",
    "high_gold_shop_access",
    "hp_ratio",
    "missing_hp_ratio",
    "hp_per_remaining_elite",
    "gold_per_remaining_shop",
    "deck_per_remaining_elite",
    "relics_per_remaining_elite",
    "distance_to_boss",
    "last_rest_distance_to_boss",
    "last_shop_distance_to_boss",
    "elites_after_last_rest",
    "monsters_after_last_rest",
    "events_after_last_rest",
    "danger_before_rest",
    "danger_after_last_rest",
    "early_danger",
    "path_len_x_act",
    "path_len_x_missing_hp",
    "path_len_x_low_hp",
    "path_len_x_mid_hp",
    "path_len_x_elite_density",
    "act_x_missing_hp",
    "act_x_elite_density",
    "act_x_monster_density",
    "ascension_x_elite_density",
    "ascension_x_missing_hp",
    "low_hp_no_rest",
    "low_hp_long_path",
    "mid_hp_long_path",
    "high_hp_long_path",
    "no_rest_long_path",
    "late_act_long_path",
    "late_act_low_hp",
    "late_act_no_rest",
    "long_path_elite_pressure",
    "long_path_monster_pressure",
    "silent_act2",
    "silent_act3",
    "ironclad_act2",
    "ironclad_act3",
    "gold_low",
    "gold_mid",
    "gold_high",
    "hp_low",
    "hp_mid",
    "hp_high",
    "player_prior_runs",
    "player_prior_win_rate",
    "player_prior_avg_floors",
    "player_experience_log",
    "player_experience_zero",
] + [f"item_hash_{idx}" for idx in range(HASH_BUCKETS)]

CATEGORICAL_FIELDS = ["character", "skill_bucket", "build_id", "game_mode"]

THRESHOLD_GROUP_SPECS = [
    ["act", "path_len_bucket", "hp_bucket"],
    ["act", "path_len_bucket"],
    ["path_len_bucket", "hp_bucket"],
    ["floor_band"],
    ["path_len_bucket"],
    ["act"],
    ["hp_bucket"],
]

KNOWN_DECISION_KEYS = {
    "ascension",
    "act",
    "floor",
    "current_hp",
    "max_hp",
    "current_gold",
    "deck",
    "relics",
    "potions",
    "candidate_path",
    "character",
    "skill_bucket",
    "victory",
    "label",
    "floors_reached",
    "run_id",
    "build_id",
    "game_mode",
    "profile_type",
    "player_count",
    "was_abandoned",
    "is_cheated",
    "player_prior_runs",
    "player_prior_win_rate",
    "player_prior_avg_floors",
    "player_experience",
}

LEAK_PATTERNS = [
    r"future",
    r"reward",
    r"later",
    r"next_card",
    r"card_choices",
    r"gold_per_floor",
    r"hp_per_floor",
    r"path_per_floor",
    r"act2",
    r"act3",
    r"boss_relic",
]

