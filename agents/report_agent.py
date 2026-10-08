from graph.state import AgentState


def report_agent(
    state: AgentState
) -> AgentState:

    ticker = state[
        "ticker"
    ]

    quant = state.get(
        "quant_analysis",
        {}
    )

    risk = state.get(
        "risk_analysis",
        {}
    )

    critic = state.get(
        "critic_result",
        {}
    )

    model_comparison = quant.get(
        "model_comparison",
        {}
    )

    finance_viable = critic.get(
        "finance_viable",
        "N/A"
    )

    finance_failure_reasons = critic.get(
        "finance_failure_reasons",
        []
    )

    warnings = critic.get(
        "warnings",
        []
    )

    cost_impact = critic.get(
        "cost_impact",
        "N/A"
    )

    prediction_bias = critic.get(
        "prediction_bias_detected",
        "N/A"
    )

    report = f"""
==================================================
        AI QUANTITATIVE RESEARCH REPORT
==================================================

Asset:
{ticker}

--------------------------------------------------
1. QUANTITATIVE ANALYSIS
--------------------------------------------------

Latest Price:
{quant.get("latest_price", "N/A")}

Average Daily Return:
{quant.get("average_daily_return", "N/A")}

20-Day Momentum:
{quant.get("momentum_20d", "N/A")}

Historical Observations:
{quant.get("number_of_observations", "N/A")}


--------------------------------------------------
2. MACHINE LEARNING ANALYSIS
--------------------------------------------------

Selected Model:
{quant.get("best_model", "N/A")}

Research Score:
{quant.get("research_score", "N/A")}

Predicted Next-Day Direction:
{quant.get("ml_direction", "N/A")}

Prediction Class:
{quant.get("ml_prediction", "N/A")}

Model Probability:
{quant.get("ml_probability", "N/A")}

Final Model Training Observations:
{quant.get("final_model_training_observations", "N/A")}


--------------------------------------------------
3. MODEL COMPARISON
--------------------------------------------------

{model_comparison}


--------------------------------------------------
4. HISTORICAL BACKTEST
--------------------------------------------------

Gross Strategy Return:
{quant.get("backtest_gross_strategy_return", "N/A")}

Net Strategy Return:
{quant.get("backtest_strategy_return", "N/A")}

Buy-and-Hold Return:
{quant.get("backtest_buy_hold_return", "N/A")}

Net Excess Return:
{quant.get("backtest_excess_return", "N/A")}

Strategy Maximum Drawdown:
{quant.get("backtest_max_drawdown", "N/A")}

Strategy Sharpe Ratio:
{quant.get("backtest_sharpe_ratio", "N/A")}

Backtest Observations:
{quant.get("backtest_observations", "N/A")}


--------------------------------------------------
5. EXECUTION COST ANALYSIS
--------------------------------------------------

Transaction Cost:
{quant.get("backtest_transaction_cost", "N/A")}

Slippage:
{quant.get("backtest_slippage", "N/A")}

Total Trading Costs:
{quant.get("backtest_total_trading_costs", "N/A")}

Position Changes:
{quant.get("backtest_position_changes", "N/A")}

Gross-to-Net Performance Difference:
{cost_impact}


--------------------------------------------------
6. RISK ANALYSIS
--------------------------------------------------

Annualized Volatility:
{risk.get("annualized_volatility", "N/A")}

Sharpe Ratio:
{risk.get("sharpe_ratio", "N/A")}

Maximum Drawdown:
{risk.get("maximum_drawdown", "N/A")}


--------------------------------------------------
7. RESEARCH VALIDATION
--------------------------------------------------

Critic Status:
{critic.get("status", "N/A")}

Finance Viability:
{finance_viable}

Prediction Bias Detected:
{prediction_bias}

Issue Count:
{critic.get("issue_count", "N/A")}

Warning Count:
{critic.get("warning_count", "N/A")}


--------------------------------------------------
8. VALIDATION ISSUES
--------------------------------------------------

{critic.get("issues", [])}


--------------------------------------------------
9. RESEARCH WARNINGS
--------------------------------------------------

{warnings}


--------------------------------------------------
10. FINANCE FAILURE REASONS
--------------------------------------------------

{finance_failure_reasons}


--------------------------------------------------
11. INTERPRETATION
--------------------------------------------------

The system evaluates candidate models using
both machine-learning performance and
quantitative financial performance.

A high classification score does not
automatically imply a useful quantitative
strategy.

The research evaluation considers:

- Classification F1 score
- Net excess return
- Backtest Sharpe ratio
- Maximum drawdown
- Transaction costs
- Slippage

The backtest uses a chronological held-out
period to reduce temporal leakage.

The final selected model is subsequently
retrained on all available labeled historical
observations before generating the next-day
research signal.

Finance viability is evaluated separately
from the research score.

Therefore, a model can have a high research
score while still being flagged as financially
non-viable if its cost-adjusted strategy does
not outperform the benchmark.


--------------------------------------------------
DISCLAIMER
--------------------------------------------------

The machine-learning output is a statistical
model prediction and is not a guaranteed
forecast of future market performance.

Historical backtest results do not guarantee
future performance.

Backtests do not fully capture real-world
effects such as market impact, liquidity
constraints, execution delays, and changing
transaction costs.

Transaction costs and slippage are configurable
research assumptions rather than actual broker
execution costs.

This system is a quantitative research
prototype and does not execute real trades.
"""

    return {
        "final_report": report
    }