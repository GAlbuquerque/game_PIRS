"""Central-bank reputation rules."""


LOW_INFLATION_OFFSET = 1.0
HIGH_INFLATION_OFFSET = 0.5
LARGE_INFLATION_DEVIATION = 6.0
BALANCED_POLICY_BAND = 2.0
REPUTATION_RECOVERY_THRESHOLD = 0.2
RECOVERY_REAL_RATE_THRESHOLD = 4.0
RECOVERY_NOMINAL_RATE_THRESHOLD = 1.0
NO_LOSS_REAL_RATE_THRESHOLD = 1.0
NO_LOSS_NOMINAL_RATE_THRESHOLD = 1.0

TARGET_RANGE_GAIN = 0.02
BALANCED_RESPONSE_GAIN = 0.01
CORRECTIVE_RESPONSE_GAIN = 0.02
WRONG_HIGH_RESPONSE_LOSS = 0.03
LARGE_WRONG_HIGH_RESPONSE_LOSS = 0.06
WRONG_LOW_RESPONSE_LOSS = 0.02
LARGE_WRONG_LOW_RESPONSE_LOSS = 0.04
BROKEN_GUIDANCE_LOSS = 0.06


def calculate_balanced_rate(
    inflation,
    target_inflation,
    unemployment,
    natural_unemployment,
    equilibrium_real_rate,
    minimum_interest_rate,
):
    """Return the feasible balanced Taylor-rule rate for the decision state."""
    raw_rate = (
        equilibrium_real_rate
        + inflation
        + 0.5 * (inflation - target_inflation)
        + 0.5 * (natural_unemployment - unemployment)
    )
    return float(max(minimum_interest_rate, raw_rate))


def update_reputation(
    current,
    inflation,
    target_inflation,
    chosen_rate,
    balanced_rate,
    *,
    selected_real_rate=None,
    broke_forward_guidance=False,
):
    """Update bounded reputation from inflation, policy stance, and promises.

    Stabilizing policy places a 0.2 floor under the period's result: this applies
    to a real rate of at least 4% above target and a nominal rate of at most 1%
    below target.
    """
    inflation_gap = inflation - target_inflation
    policy_deviation = chosen_rate - balanced_rate
    is_balanced = abs(policy_deviation) <= BALANCED_POLICY_BAND
    is_large_deviation = abs(inflation_gap) >= LARGE_INFLATION_DEVIATION

    if -LOW_INFLATION_OFFSET <= inflation_gap <= HIGH_INFLATION_OFFSET:
        delta = TARGET_RANGE_GAIN
    elif inflation_gap > HIGH_INFLATION_OFFSET:
        if policy_deviation > BALANCED_POLICY_BAND:
            delta = CORRECTIVE_RESPONSE_GAIN
        elif is_balanced:
            delta = 0.0 if is_large_deviation else BALANCED_RESPONSE_GAIN
        else:
            delta = -(
                LARGE_WRONG_HIGH_RESPONSE_LOSS
                if is_large_deviation
                else WRONG_HIGH_RESPONSE_LOSS
            )
    else:
        if policy_deviation < -BALANCED_POLICY_BAND:
            delta = CORRECTIVE_RESPONSE_GAIN
        elif is_balanced:
            delta = 0.0 if is_large_deviation else BALANCED_RESPONSE_GAIN
        else:
            delta = -(
                LARGE_WRONG_LOW_RESPONSE_LOSS
                if is_large_deviation
                else WRONG_LOW_RESPONSE_LOSS
            )

    # A clearly corrective absolute stance should not lose credibility merely
    # because it remains below/above the model's balanced-rate benchmark.
    if (
        inflation > target_inflation
        and selected_real_rate is not None
        and selected_real_rate > NO_LOSS_REAL_RATE_THRESHOLD
    ) or (
        inflation < target_inflation
        and chosen_rate < NO_LOSS_NOMINAL_RATE_THRESHOLD
    ):
        delta = max(0.0, delta)

    if broke_forward_guidance:
        delta -= BROKEN_GUIDANCE_LOSS
    minimum_reputation = 0.0
    if (
        inflation > target_inflation
        and selected_real_rate is not None
        and selected_real_rate >= RECOVERY_REAL_RATE_THRESHOLD
    ):
        minimum_reputation = REPUTATION_RECOVERY_THRESHOLD
    elif (
        inflation < target_inflation
        and chosen_rate <= RECOVERY_NOMINAL_RATE_THRESHOLD
    ):
        minimum_reputation = REPUTATION_RECOVERY_THRESHOLD
    return float(min(1.0, max(minimum_reputation, current + delta)))
