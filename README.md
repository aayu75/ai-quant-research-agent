# AI Quantitative Research Agent

A multi-agent quantitative research system for financial market analysis.

The system accepts a natural-language research query such as:

> Analyze AAPL momentum and risk

It retrieves historical market data, performs quantitative analysis,
trains and evaluates machine-learning models, runs a historical
backtest with configurable transaction costs and slippage, performs
risk analysis, validates the research through a Critic Agent, and
generates a structured research report.

The system is designed as a quantitative research prototype and does
not execute real trades.

---

## Key Features

- Multi-agent research workflow using LangGraph
- MCP-based market and quantitative tools
- Historical market data retrieval using yFinance
- Quantitative momentum and return analysis
- Machine-learning based next-day classification
- Logistic Regression and Random Forest model comparison
- Chronological train/test evaluation
- Historical strategy backtesting
- Transaction cost and slippage modeling
- Buy-and-hold benchmark comparison
- Sharpe ratio and maximum drawdown analysis
- Prediction-distribution bias detection
- Critic Agent for research validation
- Automatic retry mechanism
- Agent execution monitoring and logging
- FastAPI REST API
- Dockerized deployment
- Automated evaluation and unit tests

---

# System Architecture

```text
                         USER
                           |
                           v
                    +-------------+
                    |   FastAPI   |
                    +-------------+
                           |
                           v
                    +-------------+
                    | Agent       |
                    | Workflow    |
                    +-------------+
                           |
                           v
                    +-------------+
                    |   Planner   |
                    |    Agent    |
                    +-------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
      +-------------+             +-------------+
      | Data Agent  |             | Quant Agent |
      +-------------+             +-------------+
             |                           |
             v                           v
      +-------------+             +-------------+
      | MCP Client  |             | ML Models   |
      +-------------+             +-------------+
             |
             v
      +-------------+
      | MCP Server  |
      +-------------+
             |
             v
         yFinance

                           |
                           v
                    +-------------+
                    | Risk Agent  |
                    +-------------+
                           |
                           v
                    +-------------+
                    | Critic Agent|
                    +-------------+
                           |
                    +------+------+
                    |             |
                  PASS           FAIL
                    |             |
                    |        Retry Analysis
                    |             |
                    +------+------+
                           |
                           v
                    +-------------+
                    | Report Agent|
                    +-------------+
                           |
                           v
                    FINAL REPORT