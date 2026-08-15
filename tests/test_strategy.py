"""The closed-form race total must equal the lap-by-lap simulation exactly.

``enumerate_strategies`` ranks candidates arithmetically rather than by
projecting a lap time for each one. That is only legitimate if the two agree,
so the equivalence is tested rather than asserted in a comment: every strategy
the enumerator considers is re-simulated lap by lap and the totals compared.

Tolerance is 1e-6 s against totals of roughly 5,000 s, which is floating-point
summation order and nothing else.
"""
from __future__ import annotations

import pytest

from src.simulation import strategy as strat

FUEL = {"initial_fuel_kg": 110.0, "penalty_s_per_kg": 0.032}
PIT_LOSS_S = 25.0
BASE_LAP_S = 90.0

#: Degradation and pace offset per compound, in the shape ``deg_lookup``
#: returns. Rates span the observed range, from a Monaco-like floor to a
#: high-wear circuit.
RATES = {
    "SOFT": (0.120, 0.0, "observed"),
    "MEDIUM": (0.065, 0.4, "observed"),
    "HARD": (0.030, 0.8, "observed"),
}


def lookup(compound: str):
    """Stand in for the pipeline's degradation table."""
    return RATES[compound]


@pytest.mark.parametrize("race_laps", [44, 52, 57, 70, 78])
def test_ranking_total_matches_lap_by_lap_simulation(race_laps: int) -> None:
    """Every ranked total must reproduce the per-lap simulation."""
    results = strat.enumerate_strategies(
        race_laps, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S,
        PIT_LOSS_S, FUEL)
    assert results, f"no feasible strategy over {race_laps} laps"

    for result in results:
        simulated = strat.simulate_strategy(
            result.stints, race_laps, BASE_LAP_S, PIT_LOSS_S, FUEL)
        assert result.total_time_s == pytest.approx(
            simulated.total_time_s, abs=1e-6), (
            f"{result.label} {result.pit_laps} over {race_laps} laps")
        assert result.pit_laps == simulated.pit_laps


def brute_force(race_laps: int) -> list[strat.StrategyResult]:
    """Simulate every candidate the search considers, lap by lap.

    The reference the optimised path is measured against: no closed form, no
    pruning, no shared state with :func:`strat.enumerate_strategies` beyond
    the partition helper that defines the search space.
    """
    from itertools import product

    results = []
    for n_stints in (2, 3):
        for sequence in product(strat.DRY_COMPOUNDS, repeat=n_stints):
            if len(set(sequence)) < 2:
                continue
            for lengths in strat._split_laps(race_laps, n_stints):
                stints, start = [], 1
                for compound, length in zip(sequence, lengths):
                    deg, offset, _ = RATES[compound]
                    stints.append(strat.Stint(
                        compound=compound, start_lap=start, n_laps=length,
                        deg_rate=deg, base_offset=offset))
                    start += length
                results.append(strat.simulate_strategy(
                    stints, race_laps, BASE_LAP_S, PIT_LOSS_S, FUEL))
    results.sort(key=lambda r: r.total_time_s)
    return results


@pytest.mark.parametrize("race_laps", [44, 52, 57, 70, 78])
def test_winner_matches_exhaustive_simulation(race_laps: int) -> None:
    """The pruned, closed-form search must pick what brute force picks.

    Two things could break independently: the arithmetic could reorder
    strategies separated by less than the floating-point error, or the pruning
    could discard the winner. This checks the outcome both would corrupt.
    """
    results = strat.enumerate_strategies(
        race_laps, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S,
        PIT_LOSS_S, FUEL)
    reference = brute_force(race_laps)

    assert results[0].label == reference[0].label
    assert results[0].pit_laps == reference[0].pit_laps
    assert results[0].total_time_s == pytest.approx(
        reference[0].total_time_s, abs=1e-6)


@pytest.mark.parametrize("race_laps", [44, 52, 57, 70, 78])
def test_pruning_keeps_every_near_optimal_strategy(race_laps: int) -> None:
    """Pit windows are drawn from the near-optimal set, so none may be lost.

    A window clipped by the search rather than by its own criterion would be
    silently too narrow.
    """
    slack = 1.0
    results = strat.enumerate_strategies(
        race_laps, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S,
        PIT_LOSS_S, FUEL, near_optimal_slack_s=slack)
    reference = brute_force(race_laps)
    cutoff = reference[0].total_time_s + slack

    expected = {(r.label, tuple(r.pit_laps))
                for r in reference if r.total_time_s <= cutoff}
    kept = {(r.label, tuple(r.pit_laps)) for r in results}
    assert expected <= kept


@pytest.mark.parametrize("race_laps", [44, 52, 57, 70, 78])
def test_pruning_keeps_the_best_of_every_compound_sequence(race_laps: int) -> None:
    """The alternatives list is drawn per compound sequence, so each must
    still be represented by its own best split."""
    results = strat.enumerate_strategies(
        race_laps, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S,
        PIT_LOSS_S, FUEL)
    reference = brute_force(race_laps)

    best_per_shape: dict[tuple[str, ...], strat.StrategyResult] = {}
    for result in reference:
        shape = tuple(s.compound for s in result.stints)
        best_per_shape.setdefault(shape, result)

    # Compared on race time rather than on pit laps. Two splits of a sequence
    # that repeats a compound can tie exactly (HARD-HARD over 16 then 17 laps
    # costs what 17 then 16 costs), and which of them a sort returns first is
    # not a property worth pinning.
    kept: dict[tuple[str, ...], float] = {}
    for result in results:
        shape = tuple(s.compound for s in result.stints)
        kept[shape] = min(kept.get(shape, float("inf")), result.total_time_s)

    for shape, result in best_per_shape.items():
        assert shape in kept, shape
        assert kept[shape] == pytest.approx(result.total_time_s, abs=1e-6), shape


@pytest.mark.parametrize("race_laps", [44, 52, 57, 70, 78])
def test_pruning_discards_most_of_the_search_space(race_laps: int) -> None:
    """The point of the exercise. If this stops holding, the pruning has
    stopped doing anything and the latency figures no longer apply."""
    results = strat.enumerate_strategies(
        race_laps, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S,
        PIT_LOSS_S, FUEL)
    searched = strat.search_strategies(
        race_laps, list(strat.DRY_COMPOUNDS), RATES, BASE_LAP_S,
        PIT_LOSS_S, FUEL)
    assert len(results) < 0.25 * len(searched)


def test_only_the_returned_strategy_carries_traces() -> None:
    """Per-lap traces are the expensive part; they are built once."""
    results = strat.enumerate_strategies(
        57, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S, PIT_LOSS_S, FUEL)
    assert results[0].has_traces
    assert len(results[0].lap_times) == 57
    assert not any(r.has_traces for r in results[1:])
    assert all(r.lap_times == [] for r in results[1:])


def test_materialise_is_idempotent() -> None:
    """Materialising an already-traced strategy returns it unchanged."""
    results = strat.enumerate_strategies(
        57, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S, PIT_LOSS_S, FUEL)
    again = strat.materialise(results[0], 57, BASE_LAP_S, PIT_LOSS_S, FUEL)
    assert again is results[0]

    alternative = strat.materialise(results[1], 57, BASE_LAP_S, PIT_LOSS_S, FUEL)
    assert alternative.has_traces
    assert alternative.total_time_s == pytest.approx(
        results[1].total_time_s, abs=1e-6)


def test_fuel_cost_is_independent_of_the_stint_split() -> None:
    """The premise behind splitting the total into fixed and variable parts.

    If the fuel penalty depended on where the stops fell, the fixed term would
    not be fixed and the ranking would be wrong.
    """
    race_laps = 57
    fixed = strat.fixed_race_cost(race_laps, BASE_LAP_S, FUEL)

    for lengths in ([30, 27], [20, 37], [19, 19, 19], [25, 20, 12]):
        stints, start = [], 1
        for i, length in enumerate(lengths):
            compound = ["SOFT", "MEDIUM", "HARD"][i]
            deg, offset, _ = RATES[compound]
            stints.append(strat.Stint(compound=compound, start_lap=start,
                                      n_laps=length, deg_rate=deg,
                                      base_offset=offset))
            start += length
        simulated = strat.simulate_strategy(stints, race_laps, BASE_LAP_S,
                                            PIT_LOSS_S, FUEL)
        variable = strat.variable_race_cost(stints, PIT_LOSS_S)
        assert fixed + variable == pytest.approx(simulated.total_time_s, abs=1e-6)


def test_two_compound_rule_is_enforced() -> None:
    """A dry race must use at least two distinct compounds."""
    results = strat.enumerate_strategies(
        57, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S, PIT_LOSS_S, FUEL)
    for result in results:
        assert len({s.compound for s in result.stints}) >= 2


def test_stint_lengths_cover_the_race() -> None:
    """A partition that does not sum to the race distance breaks the fixed
    term, which assumes every strategy runs the same number of laps."""
    race_laps = 57
    results = strat.enumerate_strategies(
        race_laps, list(strat.DRY_COMPOUNDS), lookup, BASE_LAP_S,
        PIT_LOSS_S, FUEL)
    for result in results:
        assert sum(s.n_laps for s in result.stints) == race_laps
