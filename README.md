# Multi-venue-SOR-BTC-Bot

Overview
Live Smart Order Router for BTC that dynamically routes orders across multiple crypto exchanges to minimise execution cost and maximise fill rate.
Benchmarked against TWAP (Time-Weighted Average Price) and Random venue selection strategies.
Built with:

Live order book data (via CCXT REST polling)
Dynamic venue re-ranking mid-fill
Per-venue fee & latency models
Depth-aware execution (walks order book levels)
Execution quality metrics
Key Features
Multi-venue live routing: Binance, OKX, Bybit, Coinbase, Kraken
Fee & latency adjusted: per-venue basis points model
Dynamic depth consumption: re-routes between venues mid-order as marginal costs shift
Benchmark comparisons: SOR vs TWAP vs Random
Performance Metrics
We measure execution quality using three key metrics:
VWAP Slippage – How far the achieved price is from the volume-weighted average price available at order arrival.
Implementation Shortfall – Difference between the arrival mid-price and the actual execution price, showing cost vs an ideal instant fill.
Fill Rate – Percentage of the requested quantity actually filled, reflecting execution reliability.
Tech Stack
Python 3.13
CCXT for multi-exchange order book data
Pandas / Numpy for metrics
Asyncio for concurrent venue polling
