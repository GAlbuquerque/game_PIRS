# Dual-mandate calibration simulation

## Suggested end-game parameters

Use dual-mandate-specific Strong and Mixed/Poor boundaries while leaving every
inflation-target boundary unchanged:

| Mandate | Strong term loss | Strong ending loss | Mixed maximum term loss |
|---|---:|---:|---:|
| Inflation target (unchanged) | `< 2.0` | `< 1.0` | `<= 4.0` |
| Dual mandate | `< 3.0` | `< 2.0` | `<= 6.0` |

This is a transparent calibration: the dual score combines two mandate-loss
components, and the relaxed Strong boundaries bring the Good persona's second-
term Strong rate from the earlier 25% baseline to 51%, matching the inflation-
target result of 51% in the same seeded sample. Raising the dual-mandate Mixed
ceiling to 6.0 produces a 42% Mixed rate and 7% Poor rate. Strong comparisons
remain strict, while the Mixed ceiling remains inclusive.

## Simulation method

The simulation uses 100 paths per persona, 32 turns per path, two independently
scored 16-turn terms, central-banker difficulty, random initial states, and seed
`20260911`. The same run seeds are reused for each persona.

## Dual-mandate performance results

Every count is also a percentage because each row has 100 simulations.

| Persona | Term | Strong | Mixed | Poor |
|---|---:|---:|---:|---:|
| Good | 1 | 34 | 52 | 14 |
| Good | 2 | **51** | 42 | 7 |
| Hawk | 1 | 17 | 60 | 23 |
| Hawk | 2 | 33 | 56 | 11 |
| Dove | 1 | 38 | 49 | 13 |
| Dove | 2 | 48 | 38 | 14 |
| Careless | 1 | 36 | 46 | 18 |
| Careless | 2 | 23 | 32 | 45 |

With the requested 6.0 boundary, Good term 2 remains aligned on Strong, while
Mixed is four percentage points lower and Poor is four points higher than the
inflation-target result. Compared with the original 4.0 ceiling, the higher Mixed
ceiling reduces Poor results for every persona. This follows from changing the
boundary for the mandate rather than special-casing the Good persona.

## Reproduce

```bash
python scripts/simulate_persona_outcomes.py \
  --runs 100 --seed 20260911 \
  --output scripts/dual_mandate_outcomes_100x32.json
```
