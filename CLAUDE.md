# Working in this repo

Trading-strategy research and a plug-in backtester for a $5,000 retail account
(stocks, ETFs, options, crypto, prediction markets). ~13k lines of Python.

## Read the guide, not the code

Start with the short docs in `docs/guide/` and open source files only when you
need to change or debug them. Read only the doc your task needs:

| Need | Doc |
| --- | --- |
| Engine API, how a backtest runs, cost presets, writing a strategy | [docs/guide/architecture.md](docs/guide/architecture.md) |
| Which strategies exist, their params and results (keeper / no edge) | [docs/guide/strategies.md](docs/guide/strategies.md) |
| Non-`Strategy` simulators: intraday, Kalshi, Polymarket, 0DTE, pairs, ML | [docs/guide/research-modules.md](docs/guide/research-modules.md) |
| Data loaders, env vars, shared datasets in `/mnt/project-files`, blocked hosts | [docs/guide/data.md](docs/guide/data.md) |
| What each `examples/*.py` script runs | [docs/guide/examples.md](docs/guide/examples.md) |
| Running and writing tests | [docs/guide/testing.md](docs/guide/testing.md) |

Full research write-ups and summaries: [docs/README.md](docs/README.md).
Before starting a new idea, check `strategies.md` and `research-modules.md`:
many ideas have already been tested and found to have no edge.

## When you finish work

Update the guide in the same PR as the code:

- New or changed strategy: add/edit its row in `strategies.md` (file, class,
  key params, headline result, verdict).
- New module, loader, dataset or script: one line in the matching doc.
- New research write-up: a row and a short summary in `docs/README.md`.
- Keep entries to one or two lines. Docs are an index, not a copy of the code.

## Conventions

- $5,000 starting capital; realistic costs (`COSTS` presets); no look-ahead
  (signal at close t, trade at open t+1); report walk-forward or holdout
  results next to in-sample ones.
- New work branches from `main`. `pytest` must pass (offline, ~6 s).
