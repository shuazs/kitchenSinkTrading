# Testing

```bash
pip install -e ".[dev]"   # add ,ml to run the skipped ML tests
pytest -q                 # ~125 tests, ~6 s, fully offline
```

- Tests use `backtester.data.synthetic` random walks or small hand-built
  frames; they never hit the network. Keep it that way: no live downloads in
  tests.
- ML tests `pytest.importorskip` sklearn / lightgbm / torch / scipy.
- One test file per area: `test_engine.py` (fills, costs, no look-ahead,
  sweep, walk_forward), `test_new_strategies.py`, `test_crypto.py`,
  `test_fund_strategies.py`, `test_large_account.py`, `test_academic.py`,
  `test_insider.py`, `test_spreads.py`, `test_wheel.py`, `test_trump.py`,
  `test_ml.py`, `test_ml_research.py`, `test_intraday_lab.py`,
  `test_shortterm.py`, `test_pairs.py`, `test_prediction_markets.py`,
  `test_kalshi_minute.py`, `test_kalshi_catchup.py`, `test_kalshi_crowd.py`,
  `test_sentiment_overlay.py`, `test_event_markets.py`, `test_live_books.py`,
  `test_combo_search.py`.
- For a new strategy, test at least: weights shape/index, no look-ahead
  (signal at close t only affects trades from t+1), and one known-answer case
  with `CostModel(spread_bps=0, slippage_bps=0)`.
