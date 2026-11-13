r"""Price helpers: implied probability, de-vig (proportional / power), fees, contract P&L.

Fee bases (always labelled in outputs):
  tim_est_0.02     Tim's working estimate, $0.02 per $1 contract (an estimate, not a verified fee)
  kalshi_verified  Kalshi taker fee 0.07 * P * (1 - P) per contract (kalshi.com/fee-schedule; centicent rounding ignored)
Edge = model_prob - price - fee. A $1 contract bought at price p (+ fee) pays $1 if it wins.
"""
from __future__ import annotations

import math

TIM_FEE = 0.02


def TimFee(price):
    return TIM_FEE


def KalshiFee(price):
    return 0.07 * price * (1.0 - price)


FEES = {"tim_est_0.02": TimFee, "kalshi_verified": KalshiFee}


def Implied(decimal_odds):
    if decimal_odds is None or not (decimal_odds > 1.0):
        return None
    return 1.0 / decimal_odds


def DevigProportional(price_a, price_b):
    total = price_a + price_b
    return price_a / total, price_b / total


def DevigPower(price_a, price_b, tol=1e-10):
    """Find k with price_a**k + price_b**k = 1 (power method; shades favourite-longshot bias)."""
    if price_a + price_b <= 1.0:
        return DevigProportional(price_a, price_b)
    lo, hi = 1.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if price_a ** mid + price_b ** mid > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = 0.5 * (lo + hi)
    return price_a ** k, price_b ** k


def Edge(model_prob, price, fee_fn=TimFee):
    return model_prob - price - fee_fn(price)


def ContractPnl(price, won, fee_fn=TimFee):
    """P&L of one $1 contract bought at `price` plus fee. Cost = price + fee."""
    cost = price + fee_fn(price)
    return (1.0 - cost) if won else -cost, cost


def LogLoss(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -math.log(p)