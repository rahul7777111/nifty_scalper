# Fix Summary: Capture Legs at Trade Close for PDF Export

## Problem
The "Legs" column was missing from the PDF export of trades after generation because the legs value was not being captured at the time trades were closed.

## Root Cause
In `src/ui.py`, the `_pump_trades()` method handles different trade events (OPEN, UPDATE, PARTIAL_CLOSE, CLOSE). When a CLOSE event was received:
- The trade status was updated to CLOSED
- P&L values were recorded
- BUT the legs from the CLOSE event were NOT being saved to the state

This meant that when the PDF export function (`_trade_log_row_values_from_state`) tried to retrieve the legs, it would get the stale legs from the last PARTIAL_CLOSE event or OPEN event, which might have incorrect quantities or missing exit prices.

## Solution
Added a single line of code to capture the legs from the CLOSE event:

**File**: `src/ui.py`  
**Function**: `_pump_trades()` (around line 5070)  
**Change**: In the CLOSE event handler, added:
```python
state["legs"] = evt.legs
```

### Before:
```python
elif evt.event == "CLOSE":
    state["status"] = "CLOSED" if not evt.reason else f"CLOSED ({evt.reason})"
    self._note_closed_option_legs(evt)
    self._note_traded_qty(evt)
    self._note_hedge_traded_qty(evt)
```

### After:
```python
elif evt.event == "CLOSE":
    state["status"] = "CLOSED" if not evt.reason else f"CLOSED ({evt.reason})"
    # Capture legs at time of closing for PDF export
    state["legs"] = evt.legs
    self._note_closed_option_legs(evt)
    self._note_traded_qty(evt)
    self._note_hedge_traded_qty(evt)
```

## Impact
- **PDF Export**: Now includes complete legs information with exit prices for all trades, including closed trades
- **Consistency**: Legs displayed in the PDF will match what was recorded at the time of trade closing
- **No Breaking Changes**: The fix only adds data capture; it doesn't remove or modify existing functionality

## Test Verification
The fix was verified to correctly:
1. Capture legs from CLOSE events with exit prices
2. Include legs in the PDF export rows for closed trades
3. Format the legs information properly in the PDF output
