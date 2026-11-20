# gta6-trophy-rarity-forecast-01

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fadliferdian28/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fadliferdian28/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00Z
as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by mean pinball
loss on the logit scale.

## Model

Count-based decomposition. A trophy's rate is earners / players tracked:
earners only grow, while the tracked player base grows fast after launch and
dilutes rates. The forecaster reconstructs earners per trophy per read
(rate x players tracked), fits log(earners) and log(players) separately
against ln(1 + hours since launch) over the last 24 hours of reads, and
takes the rate's log-slope as the difference of the two fits — progression
minus dilution. Noisy slopes are shrunk toward zero; trophy-type priors
grounded in GTA V's long-run trophy shape (first story trophy ~42% of
players, final mission ~8%, platinum ~0.1%, 100% completion ~0.2%) cap
slopes and ceilings per type, so a noisy fit cannot send a grind trophy to
40% or the platinum anywhere. A refresh-lag shrink pulls the forecast
slightly back toward the last read, because the page at the target shows
Exophase's last refresh before it. Spreads combine a random-walk term, the
fit residual, and a refresh-jump term. When the history is perfectly flat
(no refresh seen yet), the model expects the next refresh to dilute rates,
type-dependently — early-story rates hold up best, platinum and grinds
dilute hardest — with a wide band. Every trophy's forecast is wrapped in a
fallback so the program never crashes and always writes valid output.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `.out` per
case. Standard library only, deterministic (two runs are byte-identical),
no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=effxtzzc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
