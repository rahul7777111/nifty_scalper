#!/usr/bin/env python3
"""
Test script to demonstrate directional trade scoring debug output
"""

# Simulate the debug output that would be shown when MSTOCK_DEBUG_LOG_NO_SIGNAL=1

print("=== Sample Debug Output ===")
print("[DIRECTIONAL SCORES] EMA9=24250.50 EMA21=24248.30 Diff=2.20 RSI=52.3 Bull=3 Bear=1 EMA_Up=True EMA_Down=False EMA_Neutral=False")
print("[DIRECTIONAL SCORES] EMA9=24248.75 EMA21=24249.10 Diff=-0.35 RSI=48.7 Bull=1 Bear=3 EMA_Up=False EMA_Down=True EMA_Neutral=False")
print("[DIRECTIONAL SCORES] EMA9=24249.20 EMA21=24249.15 Diff=0.05 RSI=50.1 Bull=2 Bear=2 EMA_Up=False EMA_Down=False EMA_Neutral=True")
print()

print("=== How to Interpret the Output ===")
print("EMA9/EMA21: Fast/slow EMA values")
print("Diff: EMA9 - EMA21 (positive = uptrend, negative = downtrend)")
print("RSI: Current RSI value")
print("Bull/Bear: Scoring points for bullish vs bearish signals")
print("EMA_Up/Down/Neutral: EMA trend direction")
print()

print("=== Scoring Logic ===")
print("Bullish points from:")
print("  - EMA up (+1)")
print("  - Bullish candle (+1)")
print("  - Positive spot move (+1)")
print("  - RSI >= 51 (+1)")
print("  - RSI > 50.5 (+1)")
print()
print("Bearish points from:")
print("  - EMA down (+1)")
print("  - Bearish candle (+1)")
print("  - Negative spot move (+1)")
print("  - RSI <= 49 (+1)")
print("  - RSI < 49.5 (+1)")
print()

print("=== Trade Conditions ===")
print("Bullish trade: (EMA_Up OR (EMA_Neutral AND Bull > Bear)) AND Bull >= 2")
print("Bearish trade: (EMA_Down OR (EMA_Neutral AND Bear > Bull)) AND Bear >= 2")
print()

print("To enable debug logging, set: MSTOCK_DEBUG_LOG_NO_SIGNAL=1")
print("Then run: python src/main.py")