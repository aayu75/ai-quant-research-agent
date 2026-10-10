# AI Quant Research Agent

Ask it something like "Analyze AAPL momentum and risk" and it pulls price history, trains two small models, backtests the resulting signal, checks its own numbers, and returns a written report.

This is a research prototype. It never places trades, and nothing it produces is investment advice.

The workflow is built with LangGraph and served through a small FastAPI app. Market data comes from yfinance and is exposed to the pipeline as an MCP tool, so the data layer is separate from the analysis code.

One thing to know before you read further: there are no LLM calls in the pipeline. The "agents" are plain Python functions. The planner is a regex, the analysis is pandas and scikit-learn, and the critic is a set of rules. You don't need an API key to run it, and results are reproducible for a given dataset because the models use fixed seeds.

## Quick start

You need Python 3.10 or newer and an internet connection (yfinance fetches from Yahoo Finance).

```bash
git clone https://github.com/aayu75/ai-quant-research-agent.git
cd ai-quant-research-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn api.main:app --reload
```

The API is now on http://localhost:8000, with interactive docs at `/docs`. Try it:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze AAPL momentum and risk"}'
```

The response has three fields, `query`, `ticker` and `report`, where `report` is the full text report as a single string. A run takes a few seconds, mostly spent downloading data and fitting the models.

There is no separate MCP server to start. The data agent launches `mcp_server.server` as a subprocess each time it needs prices.

### Docker

```bash
docker build -t quant-agent .
docker run -p 8000:8000 quant-agent
```

## Writing queries

The planner takes the first all-caps word of one to five letters as the ticker, skipping a short list of common words. So `Analyze MSFT risk` works and `analyze msft risk` doesn't. If nothing matches, the request fails with a 500 and an explanation.

## How it works

Every request runs through the same graph:

```
planner -> data -> quant -> risk -> critic -> report
                     ^                |
                     |---- retry <----+  (on FAIL, up to 2 times)
```

All stages read from and write to one shared state dictionary (`graph/state.py`).

- **Planner** extracts the ticker and sets up a task list.
- **Data** fetches the latest 500 daily rows for the ticker through the MCP client.
- **Quant** builds features, trains and compares the models, backtests the winner, and produces a next-day prediction.
- **Risk** computes volatility, Sharpe ratio and drawdown for the asset itself.
- **Critic** runs rule-based checks on the data, the model results and the backtest, and returns PASS or FAIL with a list of issues and warnings.
- **Report** formats everything into the final text.

If the critic fails the run, the graph loops back to the quant stage, at most twice. After that it produces the report anyway, with the validation issues listed in it.

Each agent is wrapped by `monitoring/tracker.py`, which times it and writes start, end and error events to `logs/`.

### The model step

Features are 1, 5 and 20 day returns, 20 day volatility, 5 and 20 day moving averages and their ratio, and the change in volume. The target is whether the next day's return is positive.

The data is split chronologically, with the first 80% used for training and the last 20% for testing. Two models are fit on it:

- Logistic regression behind a standard scaler
- Random forest with 200 trees and a max depth of 8

Both use balanced class weights. Each is scored on the test set, and the one with the higher research score wins. That score is a weighted blend of F1 (0.25), excess return over buy-and-hold (0.30), backtest Sharpe ratio (0.25) and a drawdown score (0.20). The winner is then refit on all the data to make the next-day prediction.

### The backtest

The strategy is long or flat. The model's signal for day *t* is applied to day *t+1*'s return, so it never trades on information it didn't have yet. Every change of position costs 0.10% in fees plus 0.05% in slippage. The result is compared with buy-and-hold over the same test period, and the report shows gross and net returns, maximum drawdown, Sharpe ratio (annualized with 252 trading days, no risk-free rate), and the number of trades.

## What the report contains

The report is plain text with these sections: quantitative analysis, machine learning analysis, model comparison, historical backtest, execution cost analysis, risk analysis, research validation, validation issues, warnings, finance failure reasons, interpretation, and a disclaimer.

## Project layout

```
api/            FastAPI app (GET /, GET /health, POST /research)
graph/          LangGraph workflow and shared state
agents/         planner, data, quant, risk, critic and report agents
ml/             features, training, model scoring, prediction
mcp_server/     MCP server and client, price and backtest tools
monitoring/     agent timing and logging
evaluation/     end-to-end evaluation harness and test cases
tools/          small shared helpers
tests/          pytest tests
```

## Tests and evaluation

```bash
pytest
```

Several tests call the real workflow, so they need network access and can fail if Yahoo Finance is slow or unavailable.

There is also an end-to-end evaluation script. It runs the three cases in `evaluation/test_cases.json` (AAPL, MSFT, NVDA) through the full graph and scores each on whether the ticker was extracted, the expected report sections are present, the numeric results are filled in, and the critic passed.

```bash
python -m evaluation.evaluator
```

Run both commands from the repository root.

## Limitations

- **One split, one regime.** Results come from a single 80/20 chronological split of about two years of daily data. Treat them as a sanity check, not evidence of an edge. Next-day direction is close to a coin flip, so don't be surprised when accuracy hovers near 50%.
- **Model selection uses the test set.** The winning model is picked on the same data the backtest reports on, which makes the reported numbers somewhat optimistic.
- **Next-day prediction alignment.** The feature builder drops the most recent row because it has no label, and the prediction step then takes the last remaining row, which is one trading day old. The prediction should be computed from the latest row instead.
- **Retries don't change anything.** The retry loop re-runs the same deterministic quant code, so a run that fails the critic will usually fail again.
- **Raw price levels as features.** The moving averages are absolute prices, which can drift out of the training range for trending stocks. The ratio feature is the more reliable one.
- **Single ticker, daily bars, no portfolio logic.** Intraday data, multiple assets and position sizing aren't supported.

## License

No license has been added yet. Until one is, all rights are reserved by the author.
