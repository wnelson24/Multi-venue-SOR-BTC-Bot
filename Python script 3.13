"""
BTC Smart Order Router (SOR) — Live books with per-venue fees & latency (no heuristic slippage)
(Python 3.13, REST via ccxt — no API keys needed for public order books)

- Dynamic SOR: refreshes live books inside the routing loop
- BTC only; prompts only for side (buy/sell) and quantity (in BTC)
- Per-venue taker fees and latency penalties
- Depth-aware paper execution (walks order book levels)
- Benchmarks: SOR vs TWAP vs Random
- Metrics: fill_rate, vwap_slippage, implementation_shortfall

First time only (same interpreter you run this in):
    pip install ccxt pandas numpy
"""

from __future__ import annotations
import asyncio as aio
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple, Optional, Callable
import pandas as pd
import ccxt  # REST polling (works on Python 3.13)

# ---------------- Settings ----------------
VENUE_IDS = ["binance", "okx", "bybit", "coinbase", "kraken"]  # add/remove if desired
BOOK_DEPTH = 25  # levels per venue to consider

# Per-venue BTC symbol mapping (Kraken uses XBT)
VENUE_SYMBOL = {
    "binance":  "BTC/USDT",
    "okx":      "BTC/USDT",
    "bybit":    "BTC/USDT",
    "coinbase": "BTC/USD",
    "kraken":   "XBT/USD",
}

# --- Fees & latency (in basis points) ---
VENUE_TAKER_BPS = {
    "binance": 1.0,
    "okx":     1.5,
    "bybit":   1.5,
    "coinbase":3.5,
    "kraken":  2.6,
}
LATENCY_BPS = {
    "binance": 0.2,
    "okx":     0.3,
    "bybit":   0.3,
    "coinbase":0.5,
    "kraken":  0.6,
}

# TWAP settings
TWAP_SLICES = 10
TWAP_INTERVAL_SEC = 0.25

# Fetch robustness
RETRIES = 3
RETRY_DELAY_SEC = 0.5

Side = Literal["buy", "sell"]

# -------------- Data Structures --------------

@dataclass
class Level:
    price: float
    size: float

@dataclass
class BookSnapshot:
    ts: float
    bids: List[Level]  # sorted high->low
    asks: List[Level]  # sorted low->high

@dataclass
class Fill:
    ts: float
    venue: str
    side: Side
    price: float
    size: float
    fee: float

@dataclass
class ExecutionReport:
    request_qty: float
    filled_qty: float
    avg_price: float
    side: Side
    fills: List[Fill]
    venue_breakdown: Dict[str, float]

# -------------- Venue Adapter (REST) --------------

class Venue:
    """REST-only venue using ccxt.fetch_order_book in a background thread."""
    def __init__(self, name: str, symbol: str, depth: int = BOOK_DEPTH):
        self.name = name
        self.symbol = symbol
        self.depth = max(1, depth)
        self.taker_fee_bps = VENUE_TAKER_BPS.get(name, 2.0)
        self.latency_bps = LATENCY_BPS.get(name, 0.0)
        self.ex = getattr(ccxt, name)()
        if hasattr(self.ex, "enableRateLimit"):
            self.ex.enableRateLimit = True
        self._markets_loaded = False

    async def _ensure_markets(self):
        if not self._markets_loaded:
            await aio.to_thread(self.ex.load_markets)
            self._markets_loaded = True

    async def watch_order_book(self) -> BookSnapshot:
        await self._ensure_markets()
        ob = await aio.to_thread(self.ex.fetch_order_book, self.symbol, self.depth)
        bids = [Level(float(p), float(s)) for p, s in ob.get("bids", [])[:self.depth]]
        asks = [Level(float(p), float(s)) for p, s in ob.get("asks", [])[:self.depth]]
        ts = ob.get("timestamp") or time.time() * 1000
        return BookSnapshot(ts=ts, bids=bids, asks=asks)

    async def close(self):
        pass  # REST no-op

# -------------- Execution Model --------------

def _walk_depth(levels: List[Level], qty: float) -> Tuple[float, float]:
    """Consume depth to fill qty; return (avg_price, filled_qty)."""
    remaining = qty
    cost = 0.0
    filled = 0.0
    for lvl in levels:
        if remaining <= 0:
            break
        take = min(remaining, max(lvl.size, 0.0))
        if take <= 0:
            continue
        cost += take * lvl.price
        filled += take
        remaining -= take
    if filled <= 0:
        return (math.nan, 0.0)
    return (cost / filled, filled)

def _reduce_levels(levels: List[Level], amount: float) -> None:
    """Reduce in-place the provided levels by amount (front-to-back)."""
    remaining = amount
    i = 0
    while remaining > 0 and i < len(levels):
        take = min(remaining, levels[i].size)
        levels[i].size -= take
        remaining -= take
        if levels[i].size <= 1e-12:
            i += 1
        else:
            break
    while levels and levels[0].size <= 1e-12:
        levels.pop(0)

class PaperSOR:
    """
    Dynamic router: after each partial fill, refreshes books (via get_books) and re-ranks venues.
    Effective price = walked-book impact + fees + latency (no heuristic slippage).
    """
    def __init__(self, venues: Dict[str, Venue]):
        self.venues = venues

    async def market(
        self,
        side: Side,
        qty: float,
        books: Dict[str, BookSnapshot],
        child_qty_hint: Optional[float] = None,
        get_books: Optional[Callable[[], aio.Future]] = None
    ) -> ExecutionReport:
        """
        child_qty_hint kept for API symmetry; not used in cost now.
        get_books: coroutine to fetch fresh books; if provided, SOR refreshes each loop.
        """
        remaining = qty
        fills: List[Fill] = []
        venue_sz: Dict[str, float] = {}

        def eff_price(v: Venue, b: BookSnapshot) -> float:
            # rank by top-of-book plus fees+latency
            if side == "buy":
                p = b.asks[0].price if b.asks else float("inf")
                total_bps = v.taker_fee_bps + v.latency_bps
                return p * (1 + total_bps / 10_000)
            else:
                p = b.bids[0].price if b.bids else 0.0
                total_bps = v.taker_fee_bps + v.latency_bps
                return p * (1 - total_bps / 10_000)

        def has_liquidity(b: BookSnapshot) -> bool:
            return bool(b.asks and b.bids)

        while remaining > 1e-10 and books:
            # pick best venue by effective top
            pick_name, pick_val = None, (float("inf") if side == "buy" else -float("inf"))
            for name, snap in books.items():
                if not has_liquidity(snap):
                    continue
                val = eff_price(self.venues[name], snap)
                if (side == "buy" and val < pick_val) or (side == "sell" and val > pick_val):
                    pick_val, pick_name = val, name
            if pick_name is None:
                break

            v = self.venues[pick_name]
            snap = books[pick_name]
            lvls = snap.asks if side == "buy" else snap.bids

            # walk venue depth for actual average price (captures impact)
            avg_px, filled = _walk_depth(lvls, remaining)
            if filled <= 0:
                books.pop(pick_name)
                if get_books is not None:
                    books = await get_books()
                    books = {k: v for k, v in books.items() if (v.bids and v.asks)}
                continue

            # Apply only fees + latency to realized avg_px
            total_bps = v.taker_fee_bps + v.latency_bps
            if side == "buy":
                adj_avg = avg_px * (1 + total_bps / 10_000)
            else:
                adj_avg = avg_px * (1 - total_bps / 10_000)

            fee_notional = avg_px * filled * (v.taker_fee_bps / 10_000)
            fills.append(Fill(ts=snap.ts, venue=v.name, side=side, price=adj_avg, size=filled, fee=fee_notional))
            venue_sz[v.name] = venue_sz.get(v.name, 0.0) + filled
            remaining -= filled
            _reduce_levels(lvls, filled)

            # Dynamic refresh: pull fresh books for next iteration
            if get_books is not None:
                books = await get_books()
                books = {k: v for k, v in books.items() if (v.bids and v.asks)}

        filled_qty = sum(f.size for f in fills)
        avg_price = sum(f.price * f.size for f in fills) / filled_qty if filled_qty > 0 else math.nan
        return ExecutionReport(qty, filled_qty, avg_price, side, fills, venue_sz)

# -------------- Metrics --------------

def compute_vwap(levels: List[Level], qty: float) -> float:
    avg, filled = _walk_depth(levels, qty)
    return avg if filled > 0 else math.nan

def implementation_shortfall(avg_px: float, arrival_mid: float, side: Side) -> float:
    if math.isnan(avg_px) or math.isnan(arrival_mid) or arrival_mid == 0:
        return math.nan
    direction = 1 if side == "buy" else -1
    return direction * (avg_px - arrival_mid) / arrival_mid

def metrics(exec_report: ExecutionReport, arrival_book: BookSnapshot, side: Side) -> Dict[str, float]:
    bench_vwap = compute_vwap(arrival_book.asks if side == "buy" else arrival_book.bids, exec_report.request_qty)
    vwap_slip = (exec_report.avg_price - bench_vwap) / bench_vwap if not math.isnan(bench_vwap) else math.nan
    bb = arrival_book.bids[0].price if arrival_book.bids else math.nan
    ba = arrival_book.asks[0].price if arrival_book.asks else math.nan
    mid = (bb + ba) / 2 if not (math.isnan(bb) or math.isnan(ba)) else math.nan
    ishort = implementation_shortfall(exec_report.avg_price, mid, side)
    return {
        "avg_price": exec_report.avg_price,
        "filled_qty": exec_report.filled_qty,
        "fill_rate": exec_report.filled_qty / exec_report.request_qty if exec_report.request_qty else 0.0,
        "vwap_slippage": vwap_slip,
        "implementation_shortfall": ishort,
    }

# -------------- Helpers / Robust Fetch --------------

def _filter_books_nonempty(books: Dict[str, BookSnapshot]) -> Dict[str, BookSnapshot]:
    """Keep only venues that actually returned liquidity."""
    return {k: v for k, v in books.items() if (v.asks and v.bids)}

async def gather_books_live(venues: Dict[str, Venue],
                            retries: int = RETRIES,
                            delay: float = RETRY_DELAY_SEC) -> Dict[str, BookSnapshot]:
    async def _one(v: Venue):
        try:
            return v.name, await v.watch_order_book()
        except Exception:
            return v.name, BookSnapshot(ts=time.time()*1000, bids=[], asks=[])
    for attempt in range(retries):
        res_list = await aio.gather(*(_one(v) for v in venues.values()))
        books = {k: v for k, v in res_list}
        nonempty = _filter_books_nonempty(books)
        if nonempty:
            return nonempty
        if attempt < retries - 1:
            await aio.sleep(delay)
    return {}

def clone_books(src: Dict[str, BookSnapshot]) -> Dict[str, BookSnapshot]:
    def clone_levels(levels: List[Level]) -> List[Level]:
        return [Level(l.price, l.size) for l in levels]
    return {k: BookSnapshot(ts=v.ts, bids=clone_levels(v.bids), asks=clone_levels(v.asks)) for k, v in src.items()}

def pick_best_venue(side: Side, books: Dict[str, BookSnapshot], venues: Dict[str, Venue]) -> str:
    """Return the best venue name by effective top-of-book after fees/latency."""
    best_name = None
    best_val = float("inf") if side == "buy" else -float("inf")
    for name, b in books.items():
        v = venues[name]
        if side == "buy":
            if not b.asks: continue
            eff = b.asks[0].price * (1 + (v.taker_fee_bps + v.latency_bps) / 10_000)
            if eff < best_val: best_val, best_name = eff, name
        else:
            if not b.bids: continue
            eff = b.bids[0].price * (1 - (v.taker_fee_bps + v.latency_bps) / 10_000)
            if eff > best_val: best_val, best_name = eff, name
    if best_name is None:
        raise RuntimeError("No liquidity available across venues")
    return best_name

# -------------- Benchmarks --------------

class TWAP:
    def __init__(self, venues: Dict[str, Venue], slices: int = TWAP_SLICES, interval_sec: float = TWAP_INTERVAL_SEC):
        self.venues = venues
        self.slices = max(1, slices)
        self.interval = max(0.0, interval_sec)

    async def run(self, side: Side, qty: float, get_books):
        slice_qty = qty / self.slices
        fills: List[Fill] = []
        venue_sz: Dict[str, float] = {}
        for _ in range(self.slices):
            books = await get_books()
            if not books:
                continue
            try:
                best_name = pick_best_venue(side, books, self.venues)
            except RuntimeError:
                continue
            part = await PaperSOR(self.venues).market(
                side, slice_qty, {best_name: books[best_name]},
                child_qty_hint=slice_qty,
                get_books=None  # TWAP locks venue per slice
            )
            for f in part.fills:
                fills.append(f)
                venue_sz[f.venue] = venue_sz.get(f.venue, 0.0) + f.size
            if self.interval > 0:
                await aio.sleep(self.interval)
        filled_qty = sum(f.size for f in fills)
        avg_price = (sum(f.price * f.size for f in fills) / filled_qty) if filled_qty > 0 else math.nan
        return ExecutionReport(qty, filled_qty, avg_price, side, fills, venue_sz)

class RandomVenue:
    def __init__(self, venues: Dict[str, Venue]):
        self.venues = venues

    async def run(self, side: Side, qty: float, get_books):
        books = await get_books()
        if not books:
            return ExecutionReport(qty, 0.0, math.nan, side, [], {})
        choice = random.choice(list(books.keys()))
        return await PaperSOR(self.venues).market(
            side, qty, {choice: books[choice]},
            child_qty_hint=qty, get_books=None
        )

# -------------- Runner --------------

async def run_once(side: Side, qty_btc: float):
    # Build venues with per-venue BTC symbol
    venues: Dict[str, Venue] = {
        name: Venue(name, VENUE_SYMBOL.get(name, "BTC/USDT"))
        for name in VENUE_IDS
    }

    async def get_books():
        return await gather_books_live(venues)

    # Initial snapshot (for metrics baseline)
    arrival_books = await get_books()
    if not arrival_books:
        raise RuntimeError("No venues returned live liquidity. Check internet or venue accessibility.")

    # Dynamic SOR: refreshes inside the loop
    sor_exec = await PaperSOR(venues).market(
        side, qty_btc, clone_books(arrival_books), child_qty_hint=qty_btc, get_books=get_books
    )
    arrival_any = next(iter(arrival_books.values()))
    sor_m = metrics(sor_exec, arrival_any, side)

    # TWAP benchmark (fresh books each slice; venue locked per slice)
    twap = TWAP(venues)
    twap_exec = await twap.run(side, qty_btc, get_books=get_books)
    twap_m = metrics(twap_exec, arrival_any, side)

    # Random benchmark (fresh books; single random venue for whole order)
    rnd = RandomVenue(venues)
    rnd_exec = await rnd.run(side, qty_btc, get_books=get_books)
    rnd_m = metrics(rnd_exec, arrival_any, side)

    # Report
    df = pd.DataFrame([
        {"strategy": "SOR", **sor_m},
        {"strategy": "TWAP", **twap_m},
        {"strategy": "Random", **rnd_m},
    ])

    def fmt_pct(x):
        return "-" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x*100:.3f}%"

    print("\n=== Execution Summary ===")
    print(f"Side: {side}  Qty: {qty_btc} BTC\nVenues: {', '.join(venues.keys())}")

    for name, er in (("SOR", sor_exec), ("TWAP", twap_exec), ("Random", rnd_exec)):
        print(f"\n{name}: filled {er.filled_qty:.6f}/{er.request_qty} @ avg {er.avg_price:.2f}")
        if er.venue_breakdown:
            parts = ", ".join([f"{k}:{v:.4f}" for k, v in er.venue_breakdown.items()])
            print(f"  venue breakdown: {parts}")

    print("\n=== Metrics (vs arrival snapshot) ===")
    for _, row in df.iterrows():
        print(f"{row['strategy']:>6} | fill_rate {row['fill_rate']:.3f} | vwap_slip {fmt_pct(row['vwap_slippage'])} | IS {fmt_pct(row['implementation_shortfall'])}")

    # Cleanup (no-op for REST)
    for v in venues.values():
        await v.close()

# -------------- Entry (minimal prompts) --------------

if __name__ == "__main__":
    side = input("Side (buy/sell): ").strip().lower()
    if side not in ("buy", "sell"):
        print("Invalid side, defaulting to 'buy'.")
        side = "buy"
    try:
        qty = float(input("Quantity (in BTC, e.g., 0.05): ").strip())
    except Exception:
        print("Invalid quantity, defaulting to 0.05 BTC.")
        qty = 0.05

    try:
        aio.run(run_once(side, qty))
    except KeyboardInterrupt:
        pass
