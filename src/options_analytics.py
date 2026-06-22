import math
import numpy as np
from datetime import date, datetime
from typing import Dict, List, Any, Optional
import logging

from greeks import implied_volatility, gamma

logger = logging.getLogger(__name__)

class OptionsAnalyticsCompiler:
    """Compiles advanced option analytics for the quantitative decision loop.
    
    Calculates Implied Volatility (IV), IV Rank, IV Percentile, PCR, GEX, and Max Pain.
    """
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.iv_history: List[float] = []
        self.spot_history: List[float] = []
        self.prev_pcr_oi: Optional[float] = None
        self.prev_oi: Dict[str, int] = {}  # key (strike_type) -> openInterest

    def calculate_metrics(self, spot: float, option_chain: List[Dict[str, Any]], rate: float = 0.07, *, as_of: Optional[date] = None) -> Dict[str, Any]:
        """Calculates advanced options metrics from the active option chain.
        
        Args:
            spot: Current spot price of the underlying index.
            option_chain: List of option contracts with bid/ask, LTP, OI, volume, strike, and expiry.
            rate: Risk-free interest rate (defaults to 7.0% for Indian market).
            
        Returns:
            Dict containing IV, IVR, IVP, PCR, GEX, Max Pain, and Volatility Regime.
        """
        metrics = {
            "atm_iv": 0.15,
            "iv_rank": 50.0,
            "iv_percentile": 50.0,
            "pcr_volume": 1.0,
            "pcr_oi": 1.0,
            "pcr_change": 0.0,
            "gamma_exposure": 0.0,
            "gamma_flip_level": spot,
            "max_pain": spot,
            "volatility_regime": "NORMAL",
            "volatility_risk_premium": 0.0,
            "dte_normalized": 0.5,
            "oi_change_ratio": 1.0,
            "put_oi_buildup": 0.0,
            "call_oi_buildup": 0.0,
            "iv_skew": 0.0,
            "vanna_proxy": 0.0,
            "charm_proxy": 0.0
        }
        
        if not option_chain or spot <= 0:
            return metrics
            
        try:
            calls = [c for c in option_chain if str(c.get("opt_type")).strip().upper() == "CE"]
            puts = [p for p in option_chain if str(p.get("opt_type")).strip().upper() == "PE"]
            
            # 1. ATM IV Calculation (average of nearest ITM/OTM Call & Put)
            atm_contracts = sorted(option_chain, key=lambda x: abs(float(x.get("strike", 0)) - spot))[:4]
            ivs = []
            days_to_expiry = 1
            for c in atm_contracts:
                ltp_raw = c.get("ltp")
                if ltp_raw is None:
                    ltp_raw = c.get("lastPrice")
                if ltp_raw is None:
                    ltp_raw = c.get("last_price")
                price = float(ltp_raw) if ltp_raw is not None else 0.0
                strike = float(c.get("strike", 0))
                opt_type = str(c.get("opt_type")).strip().upper()
                
                exp_raw = c.get("expiry")
                if not exp_raw:
                    continue
                try:
                    if isinstance(exp_raw, (date, datetime)):
                        exp_date = exp_raw if isinstance(exp_raw, date) else exp_raw.date()
                    else:
                        exp_date = datetime.strptime(str(exp_raw), "%Y-%m-%d").date()
                    valuation_date = as_of or date.today()
                    days_to_expiry = max(1, (exp_date - valuation_date).days)
                except Exception:
                    days_to_expiry = 1
                t = days_to_expiry / 365.0
                
                iv = implied_volatility(price, spot, strike, t, rate, opt_type)
                if iv and iv > 0:
                    ivs.append(iv)
            
            atm_iv = float(np.mean(ivs)) if ivs else 0.15
            metrics["atm_iv"] = atm_iv
            t = days_to_expiry / 365.0
            metrics["dte_normalized"] = days_to_expiry / 7.0
            
            # 2. IV Rank & IV Percentile
            self.iv_history.append(atm_iv)
            if len(self.iv_history) > self.lookback_days:
                self.iv_history.pop(0)
                
            min_iv = min(self.iv_history)
            max_iv = max(self.iv_history)
            
            if max_iv > min_iv:
                metrics["iv_rank"] = ((atm_iv - min_iv) / (max_iv - min_iv)) * 100.0
            else:
                metrics["iv_rank"] = 50.0
                
            smaller_ivs = sum(1 for x in self.iv_history if x < atm_iv)
            metrics["iv_percentile"] = (smaller_ivs / len(self.iv_history)) * 100.0
            
            # 3. Put-Call Ratio (PCR) & PCR Change
            total_call_vol = sum(int(c.get("volume") or 0) for c in calls)
            total_put_vol = sum(int(p.get("volume") or 0) for p in puts)
            total_call_oi = sum(int(c.get("oi") or c.get("openInterest") or 0) for c in calls)
            total_put_oi = sum(int(p.get("oi") or p.get("openInterest") or 0) for p in puts)
            
            if total_call_vol > 0:
                metrics["pcr_volume"] = float(total_put_vol) / float(total_call_vol)
            elif total_put_vol > 0:
                metrics["pcr_volume"] = float('inf')
                
            pcr_oi = 1.0
            if total_call_oi > 0:
                pcr_oi = total_put_oi / total_call_oi
            metrics["pcr_oi"] = pcr_oi
            
            if self.prev_pcr_oi is not None:
                metrics["pcr_change"] = pcr_oi - self.prev_pcr_oi
            self.prev_pcr_oi = pcr_oi
            
            # 4. Volatility Risk Premium (VRP) calculation
            self.spot_history.append(spot)
            if len(self.spot_history) > 100:
                self.spot_history.pop(0)
            if len(self.spot_history) >= 10:
                prices = np.array(self.spot_history)
                returns = np.diff(np.log(prices))
                rv = float(np.std(returns) * math.sqrt(252 * 375))
            else:
                rv = 0.15
            metrics["volatility_risk_premium"] = max(-0.1, min(0.2, atm_iv - rv))
            
            # 5. OI Change Ratio & Call/Put OI Build-ups
            total_put_oi_change = 0
            total_call_oi_change = 0
            for c in option_chain:
                strike = float(c.get("strike", 0))
                opt_type = str(c.get("opt_type")).strip().upper()
                oi = int(c.get("oi") or c.get("openInterest") or 0)
                key = f"{strike}_{opt_type}"
                
                prev_oi_val = self.prev_oi.get(key, oi)
                oi_change = oi - prev_oi_val
                self.prev_oi[key] = oi
                
                if opt_type == "PE":
                    total_put_oi_change += oi_change
                else:
                    total_call_oi_change += oi_change

            metrics["put_oi_buildup"] = float(total_put_oi_change)
            metrics["call_oi_buildup"] = float(total_call_oi_change)
            if total_call_oi_change != 0:
                metrics["oi_change_ratio"] = float(total_put_oi_change) / float(total_call_oi_change)
            else:
                metrics["oi_change_ratio"] = 1.0
            
            # 6. Option Skew (OTM Put IV - OTM Call IV)
            otm_call_strike = spot * 1.02
            otm_put_strike = spot * 0.98
            
            otm_call_contracts = [c for c in calls if float(c.get("strike", 0)) > spot]
            otm_put_contracts = [p for p in puts if float(p.get("strike", 0)) < spot]
            
            call_contract = min(otm_call_contracts, key=lambda x: abs(float(x.get("strike", 0)) - otm_call_strike)) if otm_call_contracts else None
            put_contract = min(otm_put_contracts, key=lambda x: abs(float(x.get("strike", 0)) - otm_put_strike)) if otm_put_contracts else None
            
            call_iv = atm_iv
            if call_contract:
                c_price = float(call_contract.get("ltp") or call_contract.get("lastPrice") or call_contract.get("last_price") or 0.0)
                c_strike = float(call_contract.get("strike", 0))
                call_iv = implied_volatility(c_price, spot, c_strike, t, rate, "CE") or atm_iv
                
            put_iv = atm_iv
            if put_contract:
                p_price = float(put_contract.get("ltp") or put_contract.get("lastPrice") or put_contract.get("last_price") or 0.0)
                p_strike = float(put_contract.get("strike", 0))
                put_iv = implied_volatility(p_price, spot, p_strike, t, rate, "PE") or atm_iv
                
            metrics["iv_skew"] = put_iv - call_iv
            
            # 7. Analytical Vanna and Charm Proxies (Black-Scholes approximations)
            try:
                # Standard Normal PDF helper
                def norm_pdf(x):
                    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)
                # Standard Normal CDF helper
                def norm_cdf(x):
                    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
                
                d1 = (math.log(spot / spot) + (rate + 0.5 * atm_iv * atm_iv) * t) / (atm_iv * math.sqrt(t))
                d2 = d1 - atm_iv * math.sqrt(t)
                vanna = (spot * norm_pdf(d1) * math.sqrt(t)) * (1.0 - d1 / (atm_iv * math.sqrt(t))) / spot
                charm = -norm_pdf(d1) * (rate / (atm_iv * math.sqrt(t)) - d2 / (2.0 * t))
            except Exception:
                vanna = 0.01
                charm = -0.02
                
            metrics["vanna_proxy"] = vanna
            metrics["charm_proxy"] = charm
            
            # 8. Net Gamma Exposure (GEX) & Gamma Flip Level
            total_gex = 0.0
            strike_gex = {}
            for c in option_chain:
                oi = int(c.get("oi") or c.get("openInterest") or 0)
                if oi <= 0:
                    continue
                strike = float(c.get("strike", 0))
                opt_type = str(c.get("opt_type")).strip().upper()
                price = float(c.get("ltp") or c.get("lastPrice") or c.get("last_price") or 0.0)
                
                iv = implied_volatility(price, spot, strike, t, rate, opt_type) or atm_iv
                try:
                    g = gamma(spot, strike, t, rate, iv)
                except Exception:
                    g = 0.0
                
                mult = 50.0
                gex = oi * g * mult * spot
                if opt_type == "PE":
                    gex = -gex
                total_gex += gex
                strike_gex[strike] = strike_gex.get(strike, 0.0) + gex
                
            metrics["gamma_exposure"] = total_gex
            
            # Find Gamma Flip strike
            sorted_strikes = sorted(strike_gex.keys())
            gamma_flip = spot
            for i in range(len(sorted_strikes) - 1):
                s1 = sorted_strikes[i]
                s2 = sorted_strikes[i+1]
                g1 = strike_gex[s1]
                g2 = strike_gex[s2]
                if (g1 > 0 and g2 < 0) or (g1 < 0 and g2 > 0):
                    gamma_flip = s1 - g1 * (s2 - s1) / (g2 - g1)
                    break
            metrics["gamma_flip_level"] = gamma_flip
            
            # 9. Options Max Pain calculation
            pain_strikes = {}
            unique_strikes = sorted(list(set(float(x.get("strike", 0)) for x in option_chain if float(x.get("strike", 0)) > 0)))
            
            for s_expiry in unique_strikes:
                total_pain = 0.0
                for c in option_chain:
                    oi = int(c.get("oi") or c.get("openInterest") or 0)
                    if oi <= 0:
                        continue
                    strike = float(c.get("strike", 0))
                    opt_type = str(c.get("opt_type")).strip().upper()
                    
                    if opt_type == "CE":
                        payout = max(0.0, s_expiry - strike)
                    else:
                        payout = max(0.0, strike - s_expiry)
                    total_pain += oi * payout
                pain_strikes[s_expiry] = total_pain
                
            if pain_strikes:
                metrics["max_pain"] = min(pain_strikes, key=pain_strikes.get)
                
            # 10. Volatility Regime Classification
            if atm_iv > 0.25:
                metrics["volatility_regime"] = "HIGH"
            elif atm_iv < 0.12:
                metrics["volatility_regime"] = "LOW"
            else:
                metrics["volatility_regime"] = "NORMAL"
                
        except Exception as e:
            logger.error(f"[OPTIONS ANALYTICS] Failed compilation: {e}")
            
        return metrics
