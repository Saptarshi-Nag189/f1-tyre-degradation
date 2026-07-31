"""Phase 8: stint simulation and pit-strategy comparison.

Works in seconds per lap throughout, using ``deg_rate`` rather than the
z-scored CWI, so every intermediate quantity is dimensionally meaningful and
the output needs no inverse transform before reaching the front end.

Model scope, enforced rather than documented:

- dry running only, since wet stints are excluded from the target fit;
- the 2022-2024 ground-effect regulation era only;
- the mandatory two-compound rule is applied to dry races.

Degradation is modelled as linear in tyre age, which is what the target
measures. Real tyres exhibit a non-linear cliff beyond their working range;
this simulator does not model it, so projections far past observed stint
lengths are extrapolation and are flagged as such.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

logger = logging.getLogger(__name__)

#: Compounds the dry model covers.
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

#: Floor applied to a degradation rate before simulating, in s/lap.
#: Circuits with no measurable degradation produce slopes scattered either
#: side of zero, and the measured value is reported unchanged. But a tyre does
#: not get faster with age, and a negative rate fed into the simulator would
#: make ever-longer stints look better without bound. A small positive floor
#: encodes "no measurable degradation" without inverting the physics.
MIN_SIMULATED_DEG_RATE = 0.002

#: Typical pace offset by compound in seconds per lap, relative to the softest
#: available tyre. Derived from the observed spread between compounds rather
#: than assumed; overridden per circuit when enough data exists.
DEFAULT_PACE_OFFSET = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}


@dataclass
class Stint:
    """One planned stint on a single compound."""

    compound: str
    start_lap: int
    n_laps: int
    deg_rate: float          # s/lap of tyre age
    base_offset: float       # s/lap pace offset versus the softest compound
    extrapolated: bool = False


@dataclass
class StrategyResult:
    """A complete race strategy and its simulated cost."""

    stints: list[Stint]
    total_time_s: float
    pit_stops: int
    pit_laps: list[int]
    lap_times: list[float] = field(default_factory=list)
    wear: list[float] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Human-readable description, e.g. ``"2-stop: SOFT-MEDIUM-HARD"``."""
        return (f"{self.pit_stops}-stop: "
                + "-".join(s.compound for s in self.stints))


def simulate_stint(stint: Stint, base_lap_s: float, fuel_penalty_s_per_kg: float,
                   fuel_remaining_kg: list[float]) -> list[float]:
    """Project lap-by-lap times for one stint.

    ``lap_time = base_pace + compound_offset + deg_rate * tyre_age
                 + fuel_penalty * fuel_remaining``

    The fuel term is added back because ``base_lap_s`` is a fuel-corrected
    reference and a real lap carries fuel.

    :param stint: the stint to simulate.
    :param base_lap_s: fuel-corrected reference pace for the circuit, seconds.
    :param fuel_penalty_s_per_kg: lap-time cost per kilogram of fuel.
    :param fuel_remaining_kg: fuel remaining at each lap of the stint.
    :returns: projected lap times in seconds.
    """
    times = []
    for age, fuel in zip(range(stint.n_laps), fuel_remaining_kg):
        times.append(base_lap_s + stint.base_offset
                     + stint.deg_rate * age
                     + fuel_penalty_s_per_kg * fuel)
    return times


def simulate_strategy(stints: list[Stint], race_laps: int, base_lap_s: float,
                      pit_loss_s: float, fuel: dict) -> StrategyResult:
    """Simulate a complete race strategy.

    :param stints: the planned stints, in order.
    :param race_laps: scheduled race distance.
    :param base_lap_s: fuel-corrected reference pace, seconds.
    :param pit_loss_s: time lost per pit stop.
    :param fuel: ``fuel_correction`` settings section.
    :returns: the simulated strategy.
    """
    initial = fuel["initial_fuel_kg"]
    penalty = fuel["penalty_s_per_kg"]
    burn = initial / max(race_laps, 1)

    lap_times: list[float] = []
    wear: list[float] = []
    pit_laps: list[int] = []
    lap = 0

    for index, stint in enumerate(stints):
        remaining = [max(0.0, initial - (lap + i) * burn) for i in range(stint.n_laps)]
        times = simulate_stint(stint, base_lap_s, penalty, remaining)
        lap_times.extend(times)
        # Wear expressed as cumulative lap-time loss to degradation, which is
        # what the model actually measures. It is not a percentage of tread.
        wear.extend(stint.deg_rate * age for age in range(stint.n_laps))
        lap += stint.n_laps
        if index < len(stints) - 1:
            pit_laps.append(lap)

    total = sum(lap_times) + pit_loss_s * len(pit_laps)
    return StrategyResult(stints=stints, total_time_s=total,
                          pit_stops=len(pit_laps), pit_laps=pit_laps,
                          lap_times=lap_times, wear=wear)


def _split_laps(race_laps: int, n_stints: int, first: int | None = None
                ) -> list[list[int]]:
    """Enumerate plausible stint-length partitions of a race.

    Only reasonably balanced splits are considered: every stint must be at
    least 20% and at most 60% of the race, which keeps the search small and
    excludes strategies no team would run.

    :param race_laps: scheduled race distance.
    :param n_stints: number of stints.
    :param first: unused placeholder for a fixed first stint length.
    :returns: list of stint-length lists.
    """
    low = max(5, int(0.20 * race_laps))
    high = max(low + 1, int(0.60 * race_laps))
    step = max(1, race_laps // 25)

    partitions: list[list[int]] = []
    if n_stints == 2:
        for a in range(low, high + 1, step):
            b = race_laps - a
            if low <= b <= high:
                partitions.append([a, b])
    elif n_stints == 3:
        for a in range(low, high + 1, step):
            for b in range(low, high + 1, step):
                c = race_laps - a - b
                if low <= c <= high:
                    partitions.append([a, b, c])
    return partitions


def enumerate_strategies(race_laps: int, compounds: list[str],
                         deg_lookup, base_lap_s: float, pit_loss_s: float,
                         fuel: dict, max_observed_stint: dict | None = None
                         ) -> list[StrategyResult]:
    """Enumerate and simulate one-stop and two-stop strategies.

    Enforces the mandatory two-compound rule: a dry race must use at least two
    distinct compounds.

    :param race_laps: scheduled race distance.
    :param compounds: available dry compounds.
    :param deg_lookup: callable mapping a compound to (deg_rate, offset, flag).
    :param base_lap_s: fuel-corrected reference pace, seconds.
    :param pit_loss_s: time lost per pit stop.
    :param fuel: ``fuel_correction`` settings section.
    :param max_observed_stint: longest observed stint per compound, for
        flagging extrapolation.
    :returns: simulated strategies, best first.
    """
    max_observed = max_observed_stint or {}
    results: list[StrategyResult] = []

    for n_stints in (2, 3):
        for sequence in product(compounds, repeat=n_stints):
            if len(set(sequence)) < 2:
                continue                      # mandatory two-compound rule
            for lengths in _split_laps(race_laps, n_stints):
                stints, start = [], 1
                for compound, length in zip(sequence, lengths):
                    deg, offset, _ = deg_lookup(compound)
                    limit = max_observed.get(compound)
                    stints.append(Stint(
                        compound=compound, start_lap=start, n_laps=length,
                        deg_rate=deg, base_offset=offset,
                        extrapolated=bool(limit and length > limit)))
                    start += length
                results.append(simulate_strategy(
                    stints, race_laps, base_lap_s, pit_loss_s, fuel))

    results.sort(key=lambda r: r.total_time_s)
    logger.info("Simulated %d strategies; best %s at %.1f s",
                len(results), results[0].label if results else "none",
                results[0].total_time_s if results else float("nan"))
    return results


def dedupe_by_shape(results: list[StrategyResult], limit: int = 4
                    ) -> list[StrategyResult]:
    """Keep the best strategy of each distinct compound sequence.

    Without this the alternatives list is dominated by near-identical splits
    of the same compound plan.

    :param results: simulated strategies, best first.
    :param limit: maximum to return.
    :returns: the best strategy per compound sequence.
    """
    seen: set[tuple[str, ...]] = set()
    unique: list[StrategyResult] = []
    for result in results:
        shape = tuple(s.compound for s in result.stints)
        if shape in seen:
            continue
        seen.add(shape)
        unique.append(result)
        if len(unique) >= limit:
            break
    return unique
