# Constraints — 20260213_254115_seed3721599640

- max_dice_rolls: 15
- fork_budget_per_cycle: 1
- fork_spend_mapping_required: True
- freshness_threshold_days (high-volatility): 56
- quorum_independent_receipts_for_μ: 3 (doc + measurement required)
- receipt_echo: True (2 consecutive PASS required)
- receipt_step_cap: 2 steps max (over cap → η)
- receipt_input_seal: True (input_hash required; input change invalidates)
- date_window_narrowing_on_echo_fail: True
- quiet_anchor_decay_loops: 3
- date_windows: absolute dates + timezone required (America/Phoenix)
