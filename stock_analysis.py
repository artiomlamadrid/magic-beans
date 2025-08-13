# Advanced Stock Analysis and Valuation System
# This module extends the Stock class with sophisticated financial analysis capabilities
# including DCF models, dividend discount models, hypergrowth valuations, and market analysis

from stock import Stock  # Base Stock class for data management
import numpy as np      # Numerical computing library
import pandas as pd     # Data manipulation and analysis library

class StockAnalysis(Stock):
    """
    Advanced stock analysis class that extends Stock with comprehensive valuation methods.
    
    Provides multiple valuation approaches:
    - Traditional DCF (Discounted Cash Flow) analysis
    - Dividend Discount Model (DDM) with enhanced buyback modeling
    - Hypergrowth company valuations for high-growth tech stocks
    - TAM (Total Addressable Market) based valuations
    - Platform scaling models for network effect businesses
    - Revenue multiple evolution analysis
    - Scenario analysis and sensitivity testing
    """
    
    def __init__(self, ticker, db_path="magic_beans.db", discount_rate=0.07):
        """
        Initialize StockAnalysis with enhanced analytical capabilities.
        
        Args:
            ticker (str): Stock ticker symbol
            db_path (str): Path to SQLite database
            discount_rate (float): Discount rate for DCF analysis (default 7% for conservatism)
        """
        super().__init__(ticker, db_path)
        self.discount_rate = discount_rate
        self.ticker = ticker.upper()  # Standardize ticker format
        
        # Storage for method chaining results
        self._last_dcf_result = None
        self._last_ddm_result = None
        self._last_pe_result = None
        self._last_hypergrowth_valuation = None
        self._last_data_quality = None
        self._last_export_file = None
        self._last_moving_averages = None
        self._last_scenarios = None
        self._last_results = {}  # Consolidated results storage

    # --- Getter Methods for Method Chaining Results ---
    
    def get_last_dcf(self):
        """Get the result of the last DCF analysis."""
        return self._last_dcf_result
    
    def get_last_ddm(self):
        """Get the result of the last DDM analysis."""
        return self._last_ddm_result
    
    def get_last_pe(self):
        """Get the result of the last P/E analysis."""
        return self._last_pe_result
    
    def get_last_hypergrowth(self):
        """Get the result of the last hypergrowth analysis."""
        return self._last_hypergrowth_valuation
    
    def get_last_moving_averages(self):
        """Get the result of the last moving averages calculation."""
        return self._last_moving_averages
    
    def get_last_scenarios(self):
        """Get the result of the last scenario analysis."""
        return self._last_scenarios
    
    def get_data_quality(self):
        """Get the last data quality assessment."""
        return self._last_data_quality
    
    def get_export_file(self):
        """Get the filename of the last exported analysis."""
        return self._last_export_file

    # --- Private Helper Methods for Data Analysis ---

    def _calculate_historical_fcf_growth(self):
        """
        Calculate normalized historical free cash flow growth rate from financial data.
        Implements sophisticated filtering to remove outliers and one-time events.
        
        Returns:
            float: Median FCF growth rate (capped between -5% and 15%) or None if insufficient data
        """
        try:
            cf_df = self.data.get("cashflow")
            if cf_df is None or cf_df.empty:
                return None
                
            # Find the best FCF row (highest mean with minimum threshold)
            best_row = None
            best_mean = 0
            for idx in cf_df.index:
                series = cf_df.loc[idx].dropna()
                if series.size >= 3:  # Need at least 3 data points
                    m = series.mean()
                    if m > best_mean and m > 1e9:  # Minimum $1B threshold
                        best_mean = m
                        best_row = series
                        
            if best_row is None:
                return None
                
            # Convert to chronological order and filter outliers
            values = list(best_row[::-1])
            
            # Remove potential one-time events (significant drops)
            if len(values) >= 3 and values[-1] < 0.6 * values[-2]:
                values = values[:-1]
                
            # Calculate growth rates
            growths = [(values[i] - values[i-1]) / values[i-1] 
                      for i in range(1, len(values)) if values[i-1] > 0]
            
            if not growths:
                return None
                
            # Use median for robustness and apply conservative caps
            g_med = float(np.median(growths))
            g_med = max(-0.05, min(g_med, 0.15))  # Capped at 15% for conservatism
            print(f"Normalized FCF growth (median): {g_med:.2%}")
            return g_med
        except Exception as e:
            print(f"Error deriving FCF growth: {e}")
            return None

    # --- Traditional Valuation Methods ---

    def evaluate_DDM(self, use_enhanced_model=True, custom_growth_rate=None, custom_discount_rate=None, custom_projection_years=None):
        """
        Enhanced Dividend Discount Model with buyback and capital appreciation components.
        
        Args:
            use_enhanced_model (bool): Whether to include buyback yield in valuation
            custom_growth_rate (float): Custom dividend growth rate (optional)
            custom_discount_rate (float): Custom discount rate (optional)  
            custom_projection_years (int): Custom projection period (optional)
            
        Returns:
            dict: DDM analysis results including target prices for different scenarios
        """
        try:
            div = None
            if "history" in self.data and self.data["history"] is not None and "Dividends" in self.data["history"].columns:
                dcol = self.data["history"]["Dividends"]
                div = dcol[dcol > 0]
            elif "dividends" in self.data and self.data["dividends"] is not None and not self.data["dividends"].empty:
                div = self.data["dividends"]
            if div is None or div.empty:
                print("No dividend data")
                return None

            div.index = pd.to_datetime(div.index, utc=True)
            annual = div.resample("YE").sum()
            latest_year = pd.Timestamp.utcnow().year
            ytd = div[div.index.year == latest_year]
            if 0 < len(ytd) < 4:
                est_full = ytd.sum() * (4 / len(ytd))
                annual.loc[pd.Timestamp(year=latest_year, month=12, day=31, tz='UTC')] = est_full
                print(f"Annualized current year dividends: observed {len(ytd)} payments -> ${est_full:.2f}")

            if annual.shape[0] < 3:
                print("Insufficient annual dividend history")
                return None

            annual_for_growth = annual.copy()
            if len(ytd) and len(ytd) < 4:
                annual_for_growth = annual_for_growth.iloc[:-1]
            growth_series = annual_for_growth.pct_change().dropna()
            growth_series = growth_series[(growth_series > -0.3) & (growth_series < 0.5)]
            if growth_series.empty:
                print("No stable dividend growth rates")
                return None

            median_div_growth = growth_series.median()
            last_div = annual.iloc[-1]

            if use_enhanced_model:
                info = self.data.get("info", {})
                eps = info.get("forwardEps") or info.get("trailingEps", 0)
                current_price = info.get("currentPrice", 0)
                if current_price > 0 and eps > 0:
                    earnings_growth = info.get("earningsGrowth") or info.get("revenueGrowth") or median_div_growth
                    earnings_growth = max(0.06, min(earnings_growth, 0.15))
                    
                    # Use custom parameters if provided
                    if custom_growth_rate is not None:
                        dividend_growth = custom_growth_rate
                    else:
                        dividend_growth = max(median_div_growth, min(earnings_growth * 0.8, 0.10))
                    
                    payout_ratio = info.get("payoutRatio") or min(last_div / eps, 0.7) if eps > 0 else 0.25
                    
                    # Use custom discount rate if provided
                    if custom_discount_rate is not None:
                        required_return = max(custom_discount_rate, 0.05)
                    else:
                        required_return = max(self.discount_rate, 0.065)
                    
                    # Use custom projection years if provided
                    stage1_years = custom_projection_years if custom_projection_years is not None else 7
                    stage1_growth = min(dividend_growth, 0.06)  # Lowered to 6%
                    stage2_growth = max(0.04, min(stage1_growth * 0.65, 0.04))  # Set to 4%

                    if stage1_growth >= required_return or stage2_growth >= required_return:
                        print("Growth rates too high; using traditional DDM")
                        stage1_growth = min(stage1_growth, required_return - 0.01)
                    stage2_growth = min(stage2_growth, required_return - 0.005)

                    pv_dividends = 0
                    dividend_t = last_div
                    for t in range(1, stage1_years + 1):
                        dividend_t *= (1 + stage1_growth)
                        pv_dividends += dividend_t / ((1 + required_return) ** t)
                    terminal_dividend = dividend_t * (1 + stage2_growth)
                    terminal_div_value = terminal_dividend / (required_return - stage2_growth)

                    retained_earnings_ratio = 1 - payout_ratio
                    roe_estimate = min(earnings_growth / retained_earnings_ratio, 0.20) if retained_earnings_ratio > 0 else 0.15
                    buyback_yield = info.get("dividendYield", 0) * 0.75
                    if roe_estimate > 0:
                        eps_terminal = eps * ((1 + earnings_growth) ** stage1_years)
                        terminal_pe = min(18, 1/stage2_growth * 0.85)
                        terminal_price_from_growth = eps_terminal * terminal_pe
                        pv_price_appreciation = terminal_price_from_growth / ((1 + required_return) ** stage1_years)
                        terminal_total_value = terminal_div_value + (pv_price_appreciation * 0.3) + (buyback_yield * current_price * 0.25)
                    else:
                        terminal_total_value = terminal_div_value

                    pv_terminal = terminal_total_value / ((1 + required_return) ** stage1_years)
                    enhanced_value = pv_dividends + pv_terminal
                    
                    # Calculate current yield for growth stock adjustment
                    current_yield = last_div / current_price if current_price > 0 else 0.02
                    
                    if current_yield < 0.02 and earnings_growth > 0.08:
                        total_return_multiplier = min(1.8, 1 + (earnings_growth - 0.05) * 6)  # Adjusted multiplier
                        enhanced_value *= total_return_multiplier
                        print(f"Applied total return multiplier: {total_return_multiplier:.2f}x (low yield, high growth)")

                    print(f"Enhanced DDM Analysis:")
                    print(f"  Stage 1 dividend growth: {stage1_growth:.2%} ({stage1_years} years)")
                    print(f"  Stage 2 dividend growth: {stage2_growth:.2%} (terminal)")
                    print(f"  Required return: {required_return:.2%}")
                    print(f"  PV of growing dividends: ${pv_dividends:.2f}")
                    print(f"  PV of terminal value: ${pv_terminal:.2f}")
                    print(f"  Enhanced DDM value: ${enhanced_value:.2f}")
                    return enhanced_value
                else:
                    print("Missing price/earnings data; using traditional DDM")

            g = max(0.04, min(median_div_growth, 0.06))
            discount = self.discount_rate
            if g >= discount:
                print("Dividend growth >= discount; abort DDM")
                return None
            next_div = last_div * (1 + g)
            traditional_value = next_div / (discount - g)
            print(f"Traditional DDM: last=${last_div:.2f} g={g:.2%} value=${traditional_value:.2f}")
            return traditional_value
        
        except Exception as e:
            print(f"Error in DDM evaluation: {e}")
            return None

    def evaluate_DCF(self, years=12, fade_years=8, terminal_g=0.04, forward_uplift=0.10, use_forward_margin=True, quality_adjust=True):
        """
        Comprehensive Discounted Cash Flow analysis with multiple growth phases.
        
        Args:
            years (int): Total projection period (default 12 years)
            fade_years (int): Years for growth rate to fade to terminal (default 8)
            terminal_g (float): Terminal growth rate (default 4%)
            forward_uplift (float): Forward-looking margin improvement (default 10%)
            use_forward_margin (bool): Whether to apply margin improvements
            quality_adjust (bool): Whether to apply quality-based discount rate adjustments
            
        Returns:
            dict: DCF analysis results with target prices and scenario analysis
        """
        """
        Flexible DCF with balanced assumptions.
        """
        try:
            info = self.data.get("info") or {}
            fcf_ttm = info.get("freeCashflow")
            if not fcf_ttm or fcf_ttm <= 0:
                print("No positive FCF")
                return None

            if use_forward_margin:
                rev_g = info.get("revenueGrowth") or 0.08
                fwd_fcf = fcf_ttm * (1 + min(rev_g, 0.15)) * (1 + forward_uplift)
            else:
                fwd_fcf = fcf_ttm
            print(f"Base FCF (TTM): ${fcf_ttm:,.0f} Forward-adjusted: ${fwd_fcf:,.0f}")

            hist_g = self._calculate_historical_fcf_growth()
            rev_g = info.get("revenueGrowth") or 0.08
            earn_g = info.get("earningsGrowth") or 0.08
            parts = []
            if hist_g is not None and hist_g > -0.02:
                parts.append(0.3 * hist_g)
            if rev_g > 0:
                parts.append(0.35 * rev_g)
            if earn_g > 0:
                parts.append(0.35 * earn_g)
            init_g = sum(parts) if parts else 0.08
            init_g = max(0.08, min(init_g, 0.18))  # Capped at 18%
            print(f"Initial modeled growth: {init_g:.2%}")

            discount = self.discount_rate
            if quality_adjust and info.get("marketCap", 0) > 1e12:
                discount = max(discount - 0.01, 0.06)
            if discount <= terminal_g:
                print("Invalid discount vs terminal")
                return None
            print(f"Discount rate: {discount:.2%}, Terminal growth: {terminal_g:.2%}")

            cashflows = []
            for yr in range(1, years + 1):
                if yr <= fade_years:
                    fade_frac = (yr - 1) / max(fade_years - 1, 1)
                    g_y = init_g * (1 - fade_frac) + terminal_g * fade_frac
                else:
                    g_y = terminal_g
                cf = (fwd_fcf if yr == 1 else cashflows[-1]) * (1 + g_y)
                cashflows.append(cf)
                if yr <= 5:
                    print(f"Year {yr}: g={g_y:.2%} CF=${cf:,.0f}")

            terminal_value = cashflows[-1] * (1 + terminal_g) / (discount - terminal_g)
            pv = sum(cf / ((1 + discount) ** (i - 0.5)) for i, cf in enumerate(cashflows, start=1))
            terminal_pv = terminal_value / ((1 + discount) ** (years - 0.5))
            pv += terminal_pv
            print(f"Terminal value: ${terminal_value:,.0f} PV: ${terminal_pv:,.0f}")
            print(f"Enterprise value (DCF): ${pv:,.0f}")

            exit_multiple = 22
            exit_ev_raw = cashflows[-1] * exit_multiple
            exit_ev_pv = exit_ev_raw / ((1 + discount) ** (years - 0.5))
            print(f"Exit EV check ({exit_multiple}x FCF): raw=${exit_ev_raw:,.0f} PV=${exit_ev_pv:,.0f}")

            net_cash = (info.get("totalCash") or 0) - (info.get("totalDebt") or 0)
            equity = pv + net_cash
            shares = info.get("sharesOutstanding")
            if shares:
                per_share = equity / shares
                print(f"DCF per share: ${per_share:.2f}")
                return per_share
            return equity
        except Exception as e:
            print(f"Error in DCF evaluation: {e}")
            return None

    def market_implied_growth_DCF(self, target_price=None, years=10, terminal_g=0.035):
        """
        Reverse engineer the growth rate implied by current market price.
        
        Args:
            target_price (float): Price to analyze (defaults to current price)
            years (int): Projection period (default 10 years)
            terminal_g (float): Terminal growth rate (default 3.5%)
            
        Returns:
            dict: Implied growth analysis with required growth rates
        """
        """
        Solve for initial growth rate matching target price.
        """
        try:
            info = self.data.get("info") or {}
            shares = info.get("sharesOutstanding")
            fcf = info.get("freeCashflow")
            if not shares or not fcf:
                return None
            target_price = target_price or info.get("targetMeanPrice") or info.get("regularMarketPrice") or info.get("currentPrice")
            if not target_price:
                return None
            discount = self.discount_rate
            net_cash = (info.get("totalCash") or 0) - (info.get("totalDebt") or 0)
            target_ev = target_price * shares - net_cash

            def pv_for_g(g_init):
                fade_years = 5
                cfs = []
                for yr in range(1, years + 1):
                    if yr <= fade_years:
                        fade_frac = (yr - 1) / max(fade_years - 1, 1)
                        g_y = g_init * (1 - fade_frac) + terminal_g * fade_frac
                    else:
                        g_y = terminal_g
                    cf = (fcf if yr == 1 else cfs[-1]) * (1 + g_y)
                    cfs.append(cf)
                term_val = cfs[-1] * (1 + terminal_g) / (discount - terminal_g)
                pv = sum(cf / ((1 + discount) ** (i - 0.5)) for i, cf in enumerate(cfs, start=1))
                pv += term_val / ((1 + discount) ** (years - 0.5))
                return pv

            lo, hi = 0.02, 0.35
            for _ in range(50):
                mid = (lo + hi) / 2
                if pv_for_g(mid) > target_ev:
                    hi = mid
                else:
                    lo = mid
            implied = (lo + hi) / 2
            capped = implied >= 0.349
            print(f"Market-implied initial growth ≈ {implied:.2%} (fade {years}y → {terminal_g:.2%})" +
                  (" (upper bound hit)" if capped else ""))
            return implied
        except Exception as e:
            print(f"Error solving implied growth: {e}")
            return None

    def evaluate_PE(self, use_forward=True, use_sector_premium=True):
        """
        Price-to-Earnings ratio analysis with sector and growth adjustments.
        
        Args:
            use_forward (bool): Whether to use forward P/E (default True)
            use_sector_premium (bool): Whether to apply sector-specific premiums
            
        Returns:
            dict: P/E analysis results with target price ranges
        """
        """
        Enhanced P/E with analyst adjustment.
        """
        try:
            info = self.data.get("info") or {}
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            if not price:
                return None
            trailing = info.get("trailingEps")
            forward = info.get("forwardEps") or trailing or 0.08
            eps_used = forward if (use_forward and forward and forward > 0) else trailing
            if not eps_used or eps_used <= 0:
                return None

            growth = info.get("earningsGrowth") or info.get("revenueGrowth") or 0.08
            sector = info.get("sector", "").lower()
            industry = info.get("industry", "").lower()
            market_cap = info.get("marketCap", 0)
            base_growth = max(0.05, min(growth, 0.20))  # Lowered cap to 20%

            if use_sector_premium and market_cap > 1e12:
                if "technology" in sector or "software" in industry or "cloud" in industry:
                    base_growth = min(base_growth * 1.15, 0.22)  # Lowered premium to 15%
                    print(f"Applied tech mega-cap premium: growth -> {base_growth:.2%}")

            payout = info.get("payoutRatio")
            if not payout or payout <= 0 or payout > 1:
                div = info.get("dividendRate")
                payout = min(div / trailing, 0.7) if div and trailing and trailing > 0 else 0.25

            k_base = max(self.discount_rate - 0.01, base_growth + 0.015)  # Adjusted CoE
            justified_pe = (payout * (1 + base_growth)) / (k_base - base_growth)
            if market_cap > 1e12 and base_growth > 0.10:
                quality_multiplier = min(1.25, 1 + (base_growth - 0.08) * 2.5)  # Lowered multiplier cap to 1.25x
                justified_pe *= quality_multiplier
                print(f"Applied quality multiplier: {quality_multiplier:.2f}x")

            # Adjust justified P/E using analyst target price
            target_price = info.get("targetMeanPrice") or price
            if target_price > price:
                target_pe = target_price / eps_used
                justified_pe = (justified_pe + target_pe * 0.3) / 1.3  # Blend with analyst, conservative weight
                print(f"Adjusted justified P/E with analyst target: {justified_pe:.2f}")

            current_pe = price / eps_used
            fair_value_justified = justified_pe * eps_used
            peg_ratio = 1.2 if base_growth > 0.12 else 1.0
            peg_multiple = peg_ratio * base_growth * 100
            peg_multiple = min(max(peg_multiple, 20), 50)  # Lowered cap to 50x
            fair_value_peg = peg_multiple * eps_used

            result = {
                "price": price,
                "eps_used": eps_used,
                "current_pe": current_pe,
                "growth_used": base_growth,
                "payout_used": payout,
                "discount_used": k_base,
                "justified_pe": justified_pe,
                "fair_value_justified": fair_value_justified,
                "fair_value_peg": fair_value_peg,
                "current_price": price,
                "eps": eps_used,
                "fair_value_pe": fair_value_justified
            }
            return result
        except Exception as e:
            print(f"Error in P/E evaluation: {e}")
            return None

    # --- Technical Analysis and Market Data Methods ---

    def calculate_moving_averages(self):
        """
        Calculate multiple moving averages for technical analysis.
        
        Returns:
            self: Returns self for method chaining
        """
        try:
            info = self.data.get("info", {})
            if not info:
                print("No info data available for moving averages")
                self._last_moving_averages = None
                return self
            
            # Get moving averages directly from yfinance info
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            ma_50 = info.get("fiftyDayAverage")
            ma_200 = info.get("twoHundredDayAverage")
            
            if not current_price:
                print("No current price data available")
                self._last_moving_averages = None
                return self
            
            if not ma_50 or not ma_200:
                print("Moving average data not available in yfinance info")
                self._last_moving_averages = None
                return self
            
            # Calculate percentage differences
            ma_50_pct = ((current_price - ma_50) / ma_50 * 100)
            ma_200_pct = ((current_price - ma_200) / ma_200 * 100)
            
            # Determine trend
            trend_50 = "Above" if ma_50_pct > 0 else "Below"
            trend_200 = "Above" if ma_200_pct > 0 else "Below"
            
            # Overall trend assessment
            if ma_50 > ma_200 and ma_50_pct > 0 and ma_200_pct > 0:
                overall_trend = "Strong Uptrend"
            elif ma_50 > ma_200:
                overall_trend = "Uptrend"
            elif ma_50 < ma_200 and ma_50_pct < 0 and ma_200_pct < 0:
                overall_trend = "Strong Downtrend"
            elif ma_50 < ma_200:
                overall_trend = "Downtrend"
            else:
                overall_trend = "Sideways"
            
            result = {
                "current_price": current_price,
                "ma_50": ma_50,
                "ma_200": ma_200,
                "ma_50_pct": ma_50_pct,
                "ma_200_pct": ma_200_pct,
                "trend_50": trend_50,
                "trend_200": trend_200,
                "overall_trend": overall_trend
            }
            
            self._last_moving_averages = result
            self._last_results['moving_averages'] = result
            return self
            
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            self._last_moving_averages = None
            return self
            
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            return None

    # --- Scenario Analysis and Risk Assessment ---

    def run_scenarios(self):
        """
        Run multiple valuation scenarios (optimistic, base case, pessimistic).
        
        Returns:
            dict: Scenario analysis results with probability-weighted outcomes
        """
        """
        Scenario comparison with adjusted assumptions.
        """
        scenarios = [
            ("Conservative", {"years": 10, "fade_years": 5, "terminal_g": 0.025}),
            ("Base", {"years": 12, "fade_years": 8, "terminal_g": 0.04}),
            ("Market-Like", {"years": 15, "fade_years": 10, "terminal_g": 0.045, "forward_uplift": 0.12}),
            ("Optimistic", {"years": 15, "fade_years": 10, "terminal_g": 0.045, "forward_uplift": 0.15})
        ]
        out = {}
        for name, params in scenarios:
            out[name] = self.evaluate_DCF(**params)
        print("Scenario DCF per-share:", out)
        return out

    # --- Data Validation and Quality Control ---

    def validate_data(self):
        """
        Validate the quality and completeness of financial data.
        
        Returns:
            dict: Data quality assessment with completeness scores
        """
        """Validate that essential data is available before analysis."""
        required_fields = ['info', 'cashflow']
        missing = []
        for field in required_fields:
            if field not in self.data or self.data[field] is None:
                missing.append(field)
        
        # Check for analyst recommendations (not required but useful)
        if 'analysis' not in self.data or self.data['analysis'] is None:
            print("Note: No analyst recommendations data available")
        
        if missing:
            print(f"Warning: Missing data for {', '.join(missing)}")
            return False
        return True

    # --- Analyst Consensus and Market Sentiment Analysis ---

    def parse_analyst_recommendations(self):
        """
        Parse and analyze analyst recommendations and price targets.
        
        Returns:
            dict: Processed analyst data with consensus metrics
        """
        """
        Parse analyst recommendations from analysis.csv data.
        Returns dict with consensus and scoring.
        """
        try:
            analysis_df = self.data.get("analysis")
            if analysis_df is None or analysis_df.empty:
                print("No analyst recommendations data available")
                return None
            
            # Get most recent period (0m)
            current_period = analysis_df[analysis_df['period'] == '0m']
            if current_period.empty:
                current_period = analysis_df.iloc[0:1]  # Take first row if 0m not found
            
            if current_period.empty:
                return None
            
            row = current_period.iloc[0]
            strong_buy = row.get('strongBuy', 0)
            buy = row.get('buy', 0) 
            hold = row.get('hold', 0)
            sell = row.get('sell', 0)
            strong_sell = row.get('strongSell', 0)
            
            total_analysts = strong_buy + buy + hold + sell + strong_sell
            if total_analysts == 0:
                return None
            
            # Calculate weighted sentiment score (-2 to +2)
            sentiment_score = (
                strong_buy * 2 + buy * 1 + hold * 0 + sell * (-1) + strong_sell * (-2)
            ) / total_analysts
            
            # Calculate percentages
            bullish_pct = (strong_buy + buy) / total_analysts * 100
            bearish_pct = (sell + strong_sell) / total_analysts * 100
            neutral_pct = hold / total_analysts * 100
            
            # Determine consensus
            if bullish_pct >= 60:
                consensus = "Strong Buy"
            elif bullish_pct >= 40:
                consensus = "Buy"
            elif bearish_pct >= 40:
                consensus = "Sell" 
            elif bearish_pct >= 60:
                consensus = "Strong Sell"
            else:
                consensus = "Hold/Neutral"
            
            result = {
                'total_analysts': total_analysts,
                'strong_buy': strong_buy,
                'buy': buy,
                'hold': hold,
                'sell': sell,
                'strong_sell': strong_sell,
                'sentiment_score': sentiment_score,
                'bullish_pct': bullish_pct,
                'bearish_pct': bearish_pct,
                'neutral_pct': neutral_pct,
                'consensus': consensus,
                'is_bullish': sentiment_score > 0.2,
                'is_bearish': sentiment_score < -0.2,
                'is_neutral': abs(sentiment_score) <= 0.2
            }
            
            print(f"Analyst Recommendations ({total_analysts} analysts):")
            print(f"  Strong Buy: {strong_buy} | Buy: {buy} | Hold: {hold} | Sell: {sell} | Strong Sell: {strong_sell}")
            print(f"  Bullish: {bullish_pct:.1f}% | Neutral: {neutral_pct:.1f}% | Bearish: {bearish_pct:.1f}%")
            print(f"  Consensus: {consensus} (Sentiment Score: {sentiment_score:+.2f})")
            
            return result
            
        except Exception as e:
            print(f"Error parsing analyst recommendations: {e}")
            return None

    def validate_against_analyst_consensus(self, valuations, current_price, analyst_data):
        """
        Compare model valuations against analyst consensus and validate results.
        
        Args:
            valuations (dict): Model-generated valuations
            current_price (float): Current market price
            analyst_data (dict): Processed analyst recommendations
            
        Returns:
            dict: Validation results comparing model vs. analyst consensus
        """
        """
        Validate our valuation models against analyst consensus.
        Returns alignment assessment and suggestions.
        """
        if not analyst_data or not current_price:
            return None
        
        try:
            # Calculate our average valuation
            valid_valuations = [v for v in valuations.values() if v is not None]
            if not valid_valuations:
                return None
            
            our_avg_valuation = sum(valid_valuations) / len(valid_valuations)
            our_upside = (our_avg_valuation - current_price) / current_price * 100
            
            # Compare with analyst sentiment
            analyst_bullish = analyst_data['is_bullish']
            analyst_bearish = analyst_data['is_bearish'] 
            analyst_neutral = analyst_data['is_neutral']
            
            consensus = analyst_data['consensus']
            sentiment_score = analyst_data['sentiment_score']
            
            # Determine alignment
            if analyst_bullish and our_upside > 10:
                alignment = "✅ ALIGNED: Both analysts and our models suggest BUY"
                confidence = "High"
            elif analyst_bearish and our_upside < -10:
                alignment = "✅ ALIGNED: Both analysts and our models suggest SELL"
                confidence = "High"
            elif analyst_neutral and abs(our_upside) <= 15:
                alignment = "✅ ALIGNED: Both suggest HOLD/FAIR VALUE"
                confidence = "Medium"
            elif analyst_bullish and our_upside < -10:
                alignment = "⚠️  DIVERGENT: Analysts bullish but our models bearish"
                confidence = "Low - Investigate further"
            elif analyst_bearish and our_upside > 10:
                alignment = "⚠️  DIVERGENT: Analysts bearish but our models bullish"
                confidence = "Low - Investigate further"
            else:
                alignment = "🔄 MIXED: Moderate disagreement, further analysis needed"
                confidence = "Medium"
            
            # Generate insights
            insights = []
            if abs(our_upside) > 20 and analyst_neutral:
                insights.append("Strong model signal despite neutral analyst consensus")
            if sentiment_score > 1.0 and our_upside < 0:
                insights.append("Very bullish analysts but our fundamental models suggest overvaluation")
            if sentiment_score < -1.0 and our_upside > 20:
                insights.append("Very bearish analysts but our models find significant upside")
            
            result = {
                'our_avg_valuation': our_avg_valuation,
                'our_upside_pct': our_upside,
                'analyst_consensus': consensus,
                'analyst_sentiment_score': sentiment_score,
                'alignment': alignment,
                'confidence': confidence,
                'insights': insights
            }
            
            print(f"\n{'-'*30} CONSENSUS VALIDATION {'-'*22}")
            print(f"Our Models Average: ${our_avg_valuation:.2f} ({our_upside:+.1f}%)")
            print(f"Analyst Consensus: {consensus}")
            print(f"Alignment: {alignment}")
            print(f"Confidence: {confidence}")
            if insights:
                print("Key Insights:")
                for insight in insights:
                    print(f"  • {insight}")
            
            return result
            
        except Exception as e:
            print(f"Error validating against analyst consensus: {e}")
            return None

    def data_quality_summary(self):
        """
        Provide summary of data quality for transparency.
        
        Returns:
            self: Returns self for method chaining
        """
        info = self.data.get('info', {})
        quality = {
            'market_cap': info.get('marketCap', 0) > 1e9,
            'fcf_positive': (info.get('freeCashflow', 0) or 0) > 0,
            'has_dividends': len(self.data.get('dividends', [])) > 0,
            'has_analyst_data': bool(info.get('targetMeanPrice')),
            'data_freshness': abs((pd.Timestamp.now() - pd.Timestamp.fromtimestamp(info.get('regularMarketTime', 0))).days) < 7
        }
        print(f"Data Quality: {sum(quality.values())}/5 checks passed")
        self._last_data_quality = quality
        self._last_results['data_quality'] = quality
        return self

    def export_analysis(self, filename=None):
        """
        Export complete analysis to JSON/CSV format.
        
        Args:
            filename (str): Output filename (optional, auto-generated if None)
            
        Returns:
            self: Returns self for method chaining
        """
        if not filename:
            filename = f"{self.ticker}_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.json"
        
        analysis_results = {
            'ticker': self.ticker,
            'analysis_date': pd.Timestamp.now().isoformat(),
            'current_price': self.data.get('info', {}).get('currentPrice'),
            'dcf_valuation': self.evaluate_DCF(),
            'ddm_valuation': self.evaluate_DDM(),
            'pe_analysis': self.evaluate_PE(),
            'technical_analysis': self.calculate_moving_averages(),
            'scenarios': self.run_scenarios(),
            'implied_growth': self.market_implied_growth_DCF(),
            'data_quality': getattr(self, '_last_data_quality', None)
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"Analysis exported to: {filename}")
        self._last_export_file = filename
        return self

    # --- Hypergrowth and High-Tech Stock Analysis ---

    def evaluate_hypergrowth_stock(self, use_platform_metrics=True, tam_weight=0.3, 
                                  platform_weight=0.25, revenue_multiple_weight=0.25, 
                                  exponential_dcf_weight=0.2, min_revenue_growth=0.15, 
                                  min_earnings_growth=0.20, min_market_cap=50e9):
        """
        Specialized valuation framework for hypergrowth technology companies.
        Uses multiple approaches: TAM-based, platform scaling, revenue multiple evolution.
        
        Args:
            use_platform_metrics (bool): Whether to include platform scaling analysis
            tam_weight (float): Weight for TAM-based valuation (default 0.3)
            platform_weight (float): Weight for platform scaling valuation (default 0.25)
            revenue_multiple_weight (float): Weight for revenue multiple valuation (default 0.25)
            exponential_dcf_weight (float): Weight for exponential DCF (default 0.2)
            min_revenue_growth (float): Minimum revenue growth for hypergrowth classification
            min_earnings_growth (float): Minimum earnings growth for hypergrowth classification
            min_market_cap (float): Minimum market cap for hypergrowth classification
            
        Returns:
            self: Returns self for method chaining
        """
        try:
            info = self.data.get('info', {})
            sector = info.get('sector', '').lower()
            industry = info.get('industry', '').lower()
            market_cap = info.get('marketCap', 0)
            
            # Identify if this is a hypergrowth candidate
            is_hypergrowth = self._identify_hypergrowth_company(
                min_revenue_growth=min_revenue_growth,
                min_earnings_growth=min_earnings_growth,
                min_market_cap=min_market_cap
            )
            
            if not is_hypergrowth:
                print("Not identified as hypergrowth company, using traditional DCF")
                result = self.evaluate_DCF()
                self._last_hypergrowth_valuation = result
                return self
            
            print(f"🚀 HYPERGROWTH ANALYSIS for {self.ticker}")
            print("="*50)
            
            # Method 1: Total Addressable Market (TAM) Analysis
            tam_valuation = self._tam_based_valuation()
            
            # Method 2: Platform/Network Effect Valuation
            platform_valuation = None
            if use_platform_metrics:
                platform_valuation = self._platform_scaling_valuation()
            
            # Method 3: Revenue Multiple Progression
            revenue_multiple_valuation = self._revenue_multiple_evolution()
            
            # Method 4: Exponential Growth DCF (higher growth, longer duration)
            exponential_dcf = self._exponential_growth_dcf()
            
            # Combine methods with weights
            valuations = []
            weights = []
            
            if tam_valuation:
                valuations.append(tam_valuation)
                weights.append(tam_weight)
                print(f"TAM-Based Valuation: ${tam_valuation:.2f}")
            
            if platform_valuation:
                valuations.append(platform_valuation)
                weights.append(platform_weight)
                print(f"Platform Scaling Valuation: ${platform_valuation:.2f}")
            
            if revenue_multiple_valuation:
                valuations.append(revenue_multiple_valuation)
                weights.append(revenue_multiple_weight)
                print(f"Revenue Multiple Valuation: ${revenue_multiple_valuation:.2f}")
            
            if exponential_dcf:
                valuations.append(exponential_dcf)
                weights.append(exponential_dcf_weight)
                print(f"Exponential DCF: ${exponential_dcf:.2f}")
            
            if not valuations:
                print("Unable to calculate hypergrowth valuation")
                self._last_hypergrowth_valuation = None
                return self
            
            # Weighted average
            weighted_valuation = sum(v * w for v, w in zip(valuations, weights)) / sum(weights)
            
            print(f"\n🎯 HYPERGROWTH FAIR VALUE: ${weighted_valuation:.2f}")
            self._last_hypergrowth_valuation = weighted_valuation
            return self
        
        except Exception as e:
            print(f"Error in hypergrowth evaluation: {e}")
            self._last_hypergrowth_valuation = None
            return self

    def _identify_hypergrowth_company(self, min_revenue_growth=0.15, min_earnings_growth=0.20, 
                                      min_market_cap=50e9, tech_sectors=None, growth_industries=None):
        """
        Identify whether a company qualifies as a hypergrowth technology company.
        Uses revenue growth, market cap, and sector criteria with configurable thresholds.
        
        Args:
            min_revenue_growth (float): Minimum revenue growth rate (default 0.15 = 15%)
            min_earnings_growth (float): Minimum earnings growth rate (default 0.20 = 20%)
            min_market_cap (float): Minimum market cap for platform effects (default 50B)
            tech_sectors (list): List of technology sectors to check
            growth_industries (list): List of growth industries to check
            
        Returns:
            bool: True if company qualifies as hypergrowth, False otherwise
        """
        try:
            info = self.data.get('info', {})
            
            # Set default sector and industry lists if not provided
            if tech_sectors is None:
                tech_sectors = ['technology', 'communication', 'consumer discretionary', 
                               'healthcare', 'industrials']
            if growth_industries is None:
                growth_industries = ['software', 'semiconductor', 'internet', 'streaming', 
                                   'electric', 'ai', 'cloud', 'biotechnology', 'fintech',
                                   'saas', 'platform', 'digital', 'cyber', 'automation']
            
            # Get company metrics
            revenue_growth = info.get('revenueGrowth', 0) or 0
            earnings_growth = info.get('earningsGrowth', 0) or 0
            sector = info.get('sector', '').lower()
            industry = info.get('industry', '').lower()
            market_cap = info.get('marketCap', 0) or 0
            
            # Additional growth indicators
            quarterly_revenue_growth = info.get('quarterlyRevenueGrowth', 0) or 0
            profit_margins = info.get('profitMargins', 0) or 0
            
            # Growth criteria
            high_revenue_growth = revenue_growth > min_revenue_growth
            high_earnings_growth = earnings_growth > min_earnings_growth
            high_quarterly_growth = quarterly_revenue_growth > min_revenue_growth
            
            # Sector/industry criteria
            is_tech_sector = any(sector_name in sector for sector_name in tech_sectors)
            is_growth_industry = any(ind in industry for ind in growth_industries)
            
            # Scale criteria
            significant_scale = market_cap > min_market_cap
            
            # High margin business (often indicates platform/software)
            high_margin = profit_margins > 0.15
            
            # Scoring system for more nuanced classification
            hypergrowth_score = 0
            criteria_met = []
            
            if high_revenue_growth:
                hypergrowth_score += 3
                criteria_met.append(f"Revenue growth: {revenue_growth:.1%}")
            if high_earnings_growth:
                hypergrowth_score += 3
                criteria_met.append(f"Earnings growth: {earnings_growth:.1%}")
            if high_quarterly_growth:
                hypergrowth_score += 2
                criteria_met.append(f"Quarterly growth: {quarterly_revenue_growth:.1%}")
            if is_tech_sector:
                hypergrowth_score += 2
                criteria_met.append(f"Tech sector: {info.get('sector', 'N/A')}")
            if is_growth_industry:
                hypergrowth_score += 2
                criteria_met.append(f"Growth industry: {info.get('industry', 'N/A')}")
            if significant_scale:
                hypergrowth_score += 1
                criteria_met.append(f"Market cap: ${market_cap/1e9:.1f}B")
            if high_margin:
                hypergrowth_score += 1
                criteria_met.append(f"High margins: {profit_margins:.1%}")
            
            # Threshold for hypergrowth classification (need at least 6 points)
            is_hypergrowth = hypergrowth_score >= 6
            
            if is_hypergrowth:
                print(f"✅ Classified as HYPERGROWTH company (Score: {hypergrowth_score}/11)")
                for criterion in criteria_met:
                    print(f"   ✓ {criterion}")
            else:
                print(f"❌ Not classified as hypergrowth (Score: {hypergrowth_score}/11)")
                if criteria_met:
                    print("   Criteria met:")
                    for criterion in criteria_met:
                        print(f"   • {criterion}")
            
            return is_hypergrowth
            
        except Exception as e:
            print(f"Error identifying hypergrowth: {e}")
            return False

    def _tam_based_valuation(self, default_tam_multiplier=5, market_leader_share=0.20, 
                           strong_player_share=0.10, high_margin_multiple=15, 
                           good_margin_multiple=12, low_margin_multiple=8,
                           years_to_maturity=12, discount_rate=0.12):
        """
        Total Addressable Market (TAM) based valuation for growth companies.
        Estimates company's potential based on market penetration scenarios.
        
        Args:
            default_tam_multiplier (float): Multiplier for estimating TAM from current revenue
            market_leader_share (float): Maximum market share for market leaders (default 0.20)
            strong_player_share (float): Maximum market share for strong players (default 0.10)
            high_margin_multiple (float): Revenue multiple for high margin businesses (default 15)
            good_margin_multiple (float): Revenue multiple for good margin businesses (default 12)
            low_margin_multiple (float): Revenue multiple for low margin businesses (default 8)
            years_to_maturity (int): Years to reach mature market share (default 12)
            discount_rate (float): Discount rate for uncertainty (default 0.12)
            
        Returns:
            float: TAM-based valuation per share or None if calculation fails
        """
        try:
            info = self.data.get('info', {})
            current_revenue = info.get('totalRevenue', 0)
            market_cap = info.get('marketCap', 0)
            
            if not current_revenue:
                return None
            
            # Estimate TAM based on company characteristics
            revenue_growth = info.get('revenueGrowth', 0.15)
            sector = info.get('sector', '').lower()
            industry = info.get('industry', '').lower()
            
            # Industry-specific TAM estimation logic
            tam_estimates_by_industry = {
                'artificial intelligence': 10.0,  # AI market
                'semiconductor': 5.0,             # Chip market
                'electric vehicle': 8.0,          # EV + Energy market
                'streaming': 2.0,                 # Global streaming
                'e-commerce': 6.0,                # E-commerce platform
                'cloud': 8.0,                     # Cloud computing
                'software': 4.0,                  # Enterprise software
                'biotechnology': 3.0,             # Biotech market
            }
            
            # Try to match industry to TAM estimate
            tam_trillion = None
            for industry_key, tam_value in tam_estimates_by_industry.items():
                if industry_key in industry:
                    tam_trillion = tam_value
                    break
            
            # If no specific industry match, estimate based on growth and revenue
            if tam_trillion is None:
                # Generic estimation: project revenue forward and multiply
                projected_revenue = current_revenue * (1 + revenue_growth) ** 10
                tam_trillion = projected_revenue / 1e12 * default_tam_multiplier
            
            # Determine maximum market share based on company characteristics
            profit_margin = info.get('profitMargins', 0.15)
            
            # Market leaders typically have higher margins and stronger competitive positions
            if profit_margin > 0.25 and market_cap > 500e9:  # Large, high-margin companies
                max_market_share = market_leader_share
                share_category = "market leader"
            elif profit_margin > 0.15 and market_cap > 100e9:  # Strong players
                max_market_share = (market_leader_share + strong_player_share) / 2
                share_category = "strong player"
            else:
                max_market_share = strong_player_share
                share_category = "competitive player"
            
            # Revenue at maturity
            mature_revenue = tam_trillion * 1e12 * max_market_share
            
            # Revenue multiple based on margins and business model
            if profit_margin > 0.25:
                revenue_multiple = high_margin_multiple  # High margin platforms
            elif profit_margin > 0.15:
                revenue_multiple = good_margin_multiple  # Good margin businesses
            else:
                revenue_multiple = low_margin_multiple   # Lower margin businesses
            
            # Calculate valuation
            mature_valuation = mature_revenue * revenue_multiple
            
            # Discount back to present value
            present_value = mature_valuation / ((1 + discount_rate) ** years_to_maturity)
            shares = info.get('sharesOutstanding', 1)
            
            tam_value_per_share = present_value / shares
            
            print(f"TAM Analysis:")
            print(f"  Estimated TAM: ${tam_trillion:.1f}T")
            print(f"  Target Market Share: {max_market_share:.1%} ({share_category})")
            print(f"  Mature Revenue: ${mature_revenue/1e12:.2f}T")
            print(f"  Revenue Multiple: {revenue_multiple}x")
            print(f"  Years to Maturity: {years_to_maturity}")
            print(f"  Discount Rate: {discount_rate:.1%}")
            
            return tam_value_per_share
            
        except Exception as e:
            print(f"Error in TAM valuation: {e}")
            return None

    def _platform_scaling_valuation(self, base_scaling_factor=2.0, years_high_growth=8, 
                                   years_moderate_growth=5, moderate_growth_rate=0.12,
                                   margin_expansion_factor=1.2, discount_rate=0.10, 
                                   terminal_growth=0.04):
        """
        Platform scaling valuation for network effect businesses.
        Models value creation through user/customer scaling and network effects.
        
        Args:
            base_scaling_factor (float): Base scaling multiplier for network effects (default 2.0)
            years_high_growth (int): Years of high growth with scaling benefits (default 8)
            years_moderate_growth (int): Years of moderate growth (default 5)
            moderate_growth_rate (float): Growth rate in moderate phase (default 0.12)
            margin_expansion_factor (float): Factor for margin expansion due to scale (default 1.2)
            discount_rate (float): Discount rate for valuation (default 0.10)
            terminal_growth (float): Terminal growth rate (default 0.04)
            
        Returns:
            float: Platform scaling valuation per share or None if calculation fails
        """
        try:
            info = self.data.get('info', {})
            current_revenue = info.get('totalRevenue', 0)
            current_price = info.get('currentPrice', 0)
            
            if not current_revenue or not current_price:
                return None
            
            # Determine scaling factor based on business characteristics
            industry = info.get('industry', '').lower()
            sector = info.get('sector', '').lower()
            profit_margin = info.get('profitMargins', 0.15)
            market_cap = info.get('marketCap', 0)
            
            # Adjust scaling factor based on network effect potential
            network_indicators = ['platform', 'marketplace', 'social', 'network', 'cloud', 'saas']
            has_network_effects = any(indicator in industry or indicator in sector 
                                    for indicator in network_indicators)
            
            if has_network_effects and profit_margin > 0.20:
                scaling_factor = base_scaling_factor * 1.5  # Strong network effects
            elif has_network_effects:
                scaling_factor = base_scaling_factor * 1.2  # Moderate network effects
            elif profit_margin > 0.25:
                scaling_factor = base_scaling_factor * 1.1  # High margin, potential platform
            else:
                scaling_factor = base_scaling_factor  # Standard scaling
            
            # Current revenue growth rate
            revenue_growth = info.get('revenueGrowth', 0.20)
            
            # Project cash flows with platform scaling
            future_revenues = []
            revenue = current_revenue
            
            # Phase 1: High growth with scaling benefits
            for year in range(1, years_high_growth + 1):
                # Growth rate decreases over time but enhanced by scaling
                growth_rate = revenue_growth * (scaling_factor / year) ** 0.3
                growth_rate = max(growth_rate, moderate_growth_rate)  # Floor at moderate growth
                revenue *= (1 + growth_rate)
                future_revenues.append(revenue)
            
            # Phase 2: Moderate growth  
            for year in range(years_moderate_growth):
                revenue *= (1 + moderate_growth_rate)
                future_revenues.append(revenue)
            
            # Convert to FCF with margin expansion due to scale
            current_fcf_margin = profit_margin * 0.8  # Approximate FCF margin from profit margin
            target_fcf_margin = min(current_fcf_margin * margin_expansion_factor, 0.35)  # Cap at 35%
            
            future_fcfs = []
            for i, rev in enumerate(future_revenues):
                # Gradually improve margins due to scale effects
                years_elapsed = i + 1
                total_years = len(future_revenues)
                margin_progress = years_elapsed / total_years
                
                fcf_margin = current_fcf_margin + (target_fcf_margin - current_fcf_margin) * margin_progress
                future_fcfs.append(rev * fcf_margin)
            
            # Terminal value
            terminal_fcf = future_fcfs[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            
            # NPV calculation
            pv_fcfs = sum(fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(future_fcfs))
            pv_terminal = terminal_value / ((1 + discount_rate) ** len(future_fcfs))
            
            total_value = pv_fcfs + pv_terminal
            shares = info.get('sharesOutstanding', 1)
            
            platform_value_per_share = total_value / shares
            
            print(f"Platform Scaling Analysis:")
            print(f"  Scaling Factor: {scaling_factor:.1f}x")
            print(f"  Network Effects: {'Yes' if has_network_effects else 'No'}")
            print(f"  High Growth Years: {years_high_growth}")
            print(f"  Current FCF Margin: {current_fcf_margin:.1%}")
            print(f"  Target FCF Margin: {target_fcf_margin:.1%}")
            print(f"  Discount Rate: {discount_rate:.1%}")
            
            return platform_value_per_share
            
        except Exception as e:
            print(f"Error in platform valuation: {e}")
            return None

    def _revenue_multiple_evolution(self, years_forward=3, growth_deceleration=0.9, 
                                   hypergrowth_multiple=25, strong_growth_multiple=18,
                                   moderate_growth_multiple=12, mature_growth_multiple=8,
                                   discount_rate=0.11):
        """
        Revenue multiple evolution analysis for high-growth companies.
        Models how revenue multiples change as companies mature.
        
        Args:
            years_forward (int): Years to project forward (default 3)
            growth_deceleration (float): Annual growth deceleration factor (default 0.9)
            hypergrowth_multiple (float): P/S multiple for >30% growth (default 25)
            strong_growth_multiple (float): P/S multiple for >20% growth (default 18)
            moderate_growth_multiple (float): P/S multiple for >10% growth (default 12)
            mature_growth_multiple (float): P/S multiple for <10% growth (default 8)
            discount_rate (float): Discount rate for present value (default 0.11)
            
        Returns:
            float: Revenue multiple valuation per share or None if calculation fails
        """
        try:
            info = self.data.get('info', {})
            current_revenue = info.get('totalRevenue', 0)
            price_to_sales = info.get('priceToSalesTrailing12Months', 0)
            
            if not current_revenue or not price_to_sales:
                return None
            
            # Current revenue growth rate
            revenue_growth = info.get('revenueGrowth', 0.15)
            
            # Determine target multiple based on growth profile
            if revenue_growth > 0.30:
                target_multiple = hypergrowth_multiple
                growth_category = "hypergrowth"
            elif revenue_growth > 0.20:
                target_multiple = strong_growth_multiple
                growth_category = "strong growth"
            elif revenue_growth > 0.10:
                target_multiple = moderate_growth_multiple
                growth_category = "moderate growth"
            else:
                target_multiple = mature_growth_multiple
                growth_category = "mature growth"
        
            # Project forward revenue with deceleration
            forward_revenue = current_revenue
            
            for year in range(years_forward):
                # Apply growth deceleration each year
                adjusted_growth = revenue_growth * (growth_deceleration ** year)
                forward_revenue *= (1 + adjusted_growth)
        
            # Apply target multiple to forward revenue
            forward_valuation = forward_revenue * target_multiple
            
            # Discount back to present value
            present_valuation = forward_valuation / ((1 + discount_rate) ** years_forward)
            
            shares = info.get('sharesOutstanding', 1)
            multiple_value_per_share = present_valuation / shares
            
            print(f"Revenue Multiple Analysis:")
            print(f"  Current P/S: {price_to_sales:.1f}x")
            print(f"  Target P/S: {target_multiple:.1f}x ({growth_category})")
            print(f"  Forward Revenue: ${forward_revenue/1e9:.1f}B")
            print(f"  Years Forward: {years_forward}")
            print(f"  Growth Deceleration: {growth_deceleration:.1%}/year")
            
            return multiple_value_per_share
        
        except Exception as e:
            print(f"Error in revenue multiple valuation: {e}")
            return None

    def _exponential_growth_dcf(self, years=15, fade_years=10, terminal_g=0.05, 
                               forward_uplift=0.25, initial_growth_multiplier=2.0, 
                               max_growth_cap=0.50):
        """
        Exponential growth DCF model for very high-growth companies.
        Uses exponential decay functions to model growth rate evolution.
        
        Args:
            years (int): Total projection period (default 15)
            fade_years (int): Years for growth fade (default 10)
            terminal_g (float): Terminal growth rate (default 0.05)
            forward_uplift (float): Forward-looking uplift factor (default 0.25)
            initial_growth_multiplier (float): Multiplier for base growth (default 2.0)
            max_growth_cap (float): Maximum growth rate cap (default 0.50)
            
        Returns:
            float: Exponential growth DCF valuation or None if calculation fails
        """
        try:
            info = self.data.get('info', {})
            revenue_growth = info.get('revenueGrowth', 0.15)
            earnings_growth = info.get('earningsGrowth', 0.15)
            
            # Use higher of the two growth rates, with multiplier and cap
            base_growth = max(revenue_growth, earnings_growth)
            exponential_growth = min(base_growth * initial_growth_multiplier, max_growth_cap)
            
            # Store original method and temporarily override
            original_method = self._calculate_historical_fcf_growth
            self._calculate_historical_fcf_growth = lambda: exponential_growth
            
            # Build parameters dictionary, excluding the multiplier
            exponential_params = {
                "years": years,
                "fade_years": fade_years, 
                "terminal_g": terminal_g,
                "forward_uplift": forward_uplift
            }
            
            result = self.evaluate_DCF(**exponential_params)
            
            # Restore original method
            self._calculate_historical_fcf_growth = original_method
            
            print(f"Exponential DCF:")
            print(f"  Initial Growth: {exponential_growth:.1%}")
            print(f"  Growth Multiplier: {initial_growth_multiplier}x")
            print(f"  Growth Years: {years}")
            print(f"  Fade Years: {fade_years}")
            
            return result
        
        except Exception as e:
            print(f"Error in exponential DCF: {e}")
            return None

if __name__ == "__main__":
    import sys
    import random
    import time
    
    # Comprehensive stock testing pools
    hypergrowth_stocks = ["NVDA", "TSLA", "NFLX", "SHOP"]
    
    # 50 diverse stocks from NASDAQ and NYSE for comprehensive testing
    test_universe = [
        # Large Cap Tech (NASDAQ)
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
        "ADBE", "CRM", "INTC", "AMD", "ORCL", "CSCO", "AVGO", "QCOM",
        
        # Large Cap Traditional (NYSE)
        "JPM", "JNJ", "PG", "UNH", "HD", "V", "MA", "DIS", "WMT", "PFE",
        "XOM", "CVX", "KO", "MRK", "BAC", "WFC", "T", "VZ",
        
        # Mid/Small Cap Growth (Mixed)
        "SHOP", "SQ", "ROKU", "ZOOM", "DOCU", "SNOW", "PLTR", "RBLX",
        "U", "DDOG", "CRWD", "ZS", "OKTA", "MDB", "TWLO",
        
        # Diverse Sectors
        "BA", "CAT", "DE", "GE", "F", "GM", "COST", "TGT", "NKE", "SBUX",
        "AMT", "CCI", "SPG", "O", "REYN"  # REITs and others
    ]
    
    # --- Standalone Analysis Functions for Portfolio and Market Research ---

def comprehensive_stock_analysis(ticker):
    """
    Perform comprehensive analysis on a single stock using all available methods.
    This function orchestrates the complete analytical workflow.
    
    Args:
        ticker (str): Stock ticker symbol to analyze
        
    Returns:
        dict: Complete analysis results including all valuation methods and quality scores
    """
    try:
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE ANALYSIS: {ticker}")
        print(f"{'='*70}")
        
        stock = StockAnalysis(ticker)
        
        # Load data with fallback to API
        data_loaded = False
        try:
            stock.load_info_from_file()
            stock.load_analysis_from_file()
            stock.load_financials_from_file()
            stock.load_history_from_file()
            
            info = stock.data.get('info', {})
            if info.get('currentPrice', 0) > 0:
                data_loaded = True
                print(f"✓ Loaded from files")
            else:
                print("Files incomplete, fetching fresh data...")
        except Exception as e:
            print(f"Loading from files failed: {e}")
        
        if not data_loaded:
            try:
                print("Fetching fresh data...")
                stock.fetch_data()
                stock.fetch_analysis()
                stock.fetch_financials()
                stock.fetch_history()
                print("✓ Fresh data fetched")
            except Exception as e:
                print(f"❌ Failed to fetch data: {e}")
                return None
        
        # Basic company info
        info = stock.data.get('info', {})
        current_price = info.get('currentPrice', 0)
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        if not current_price:
            print(f"❌ No price data for {ticker}")
            return None
        
        print(f"\n📊 COMPANY OVERVIEW")
        print(f"Price: ${current_price:.2f} | Market Cap: ${market_cap/1e9:.1f}B")
        print(f"Sector: {sector} | Industry: {industry}")
        
        results = {
            'ticker': ticker,
            'current_price': current_price,
            'market_cap': market_cap,
            'sector': sector,
            'industry': industry,
            'analysis_success': True
        }
        
        # Determine analysis approach
        is_hypergrowth = stock._identify_hypergrowth_company()
        results['is_hypergrowth'] = is_hypergrowth
        
        # Primary valuation method
        if is_hypergrowth:
            print(f"\n🚀 HYPERGROWTH FRAMEWORK")
            print("-" * 30)
            primary_value = stock.evaluate_hypergrowth_stock()
            results['primary_method'] = 'hypergrowth'
        else:
            print(f"\n📈 TRADITIONAL DCF")
            print("-" * 30)
            primary_value = stock.evaluate_DCF()
            results['primary_method'] = 'dcf'
        
        results['primary_value'] = primary_value
        
        # Secondary valuations
        print(f"\n💰 SECONDARY VALUATIONS")
        print("-" * 30)
        
        # DDM (if applicable)
        ddm_value = stock.evaluate_DDM()
        results['ddm_value'] = ddm_value
        
        # P/E Analysis
        pe_result = stock.evaluate_PE()
        pe_value = pe_result.get('fair_value_justified') if pe_result else None
        results['pe_value'] = pe_value
        results['current_pe'] = pe_result.get('current_pe') if pe_result else None
        
        # Technical Analysis
        tech_result = stock.calculate_moving_averages()
        if tech_result:
            results['trend'] = tech_result.get('overall_trend')
            results['ma_50_pct'] = tech_result.get('ma_50_pct')
            results['ma_200_pct'] = tech_result.get('ma_200_pct')
        
        # Analyst Consensus
        print(f"\n📈 ANALYST CONSENSUS")
        print("-" * 30)
        analyst_data = stock.parse_analyst_recommendations()
        if analyst_data:
            results['analyst_consensus'] = analyst_data.get('consensus')
            results['analyst_score'] = analyst_data.get('sentiment_score')
            results['analyst_bullish_pct'] = analyst_data.get('bullish_pct')
            
            # Validate against consensus
            valuations = {}
            if primary_value: valuations[results['primary_method']] = primary_value
            if pe_value: valuations['pe'] = pe_value
            if ddm_value: valuations['ddm'] = ddm_value
            
            if valuations:
                validation = stock.validate_against_analyst_consensus(
                    valuations, current_price, analyst_data
                )
                if validation:
                    results['consensus_alignment'] = validation.get('alignment', '')
                    results['model_upside'] = validation.get('our_upside_pct', 0)
        
        # Summary metrics
        if primary_value:
            upside = (primary_value - current_price) / current_price * 100
            results['primary_upside'] = upside
            print(f"\n🎯 VALUATION SUMMARY")
            print(f"Primary ({results['primary_method'].upper()}): ${primary_value:.2f} ({upside:+.1f}%)")
            if pe_value:
                pe_upside = (pe_value - current_price) / current_price * 100
                print(f"P/E Valuation: ${pe_value:.2f} ({pe_upside:+.1f}%)")
            if ddm_value:
                ddm_upside = (ddm_value - current_price) / current_price * 100
                print(f"DDM Valuation: ${ddm_value:.2f} ({ddm_upside:+.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing {ticker}: {e}")
        return {
            'ticker': ticker,
            'analysis_success': False,
            'error': str(e)
        }

def run_comprehensive_test_suite(sample_size=50):
    """
    Run comprehensive analysis on a sample of stocks for model validation.
    This function was used to validate the analysis framework across markets.
    
    Args:
        sample_size (int): Number of stocks to analyze (default 50)
        
    Returns:
        dict: Aggregated results across all analyzed stocks with performance metrics
    """
    import random
    
    # Sample universe of stocks for testing
    test_universe = [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'AVGO', 'LLY',
        'JPM', 'UNH', 'XOM', 'V', 'PG', 'MA', 'JNJ', 'HD', 'CVX', 'ABBV',
        'PFE', 'KO', 'ORCL', 'BAC', 'ASML', 'MRK', 'COST', 'CRM', 'TMO', 'WMT',
        'ADBE', 'NFLX', 'ACN', 'AMD', 'DIS', 'LIN', 'VZ', 'NKE', 'ABT', 'CSCO',
        'TXN', 'QCOM', 'DHR', 'WFC', 'BMY', 'PM', 'RTX', 'AMGN', 'SPGI', 'NOW'
    ]
    
    print(f"🔬 COMPREHENSIVE MODEL TESTING")
    print(f"Testing {sample_size} stocks from our universe...")
    print("="*80)
    
    # Randomly sample from our test universe
    sample_stocks = random.sample(test_universe, min(sample_size, len(test_universe)))
    
    all_results = []
    successful_analyses = 0
    hypergrowth_count = 0
    traditional_count = 0
    
    for i, ticker in enumerate(sample_stocks, 1):
        print(f"\n[{i}/{len(sample_stocks)}] Testing {ticker}...")
        
        result = comprehensive_stock_analysis(ticker)
        if result:
            all_results.append(result)
            
            if result.get('analysis_success', False):
                successful_analyses += 1
                if result.get('is_hypergrowth', False):
                    hypergrowth_count += 1
                else:
                    traditional_count += 1
        
        # Brief pause to avoid rate limiting
            import time
            time.sleep(2)
        
        # Generate comprehensive summary
        print(f"\n\n📊 COMPREHENSIVE TEST RESULTS")
        print("="*80)
        print(f"Total Stocks Tested: {len(sample_stocks)}")
        print(f"Successful Analyses: {successful_analyses}")
        print(f"Hypergrowth Framework Used: {hypergrowth_count}")
        print(f"Traditional Framework Used: {traditional_count}")
        
        if successful_analyses > 0:
            # Analyze results by category
            successful_results = [r for r in all_results if r.get('analysis_success', False)]
            
            # Upside distribution
            upsides = [r.get('primary_upside', 0) for r in successful_results if r.get('primary_upside') is not None]
            if upsides:
                avg_upside = sum(upsides) / len(upsides)
                positive_calls = len([u for u in upsides if u > 10])
                negative_calls = len([u for u in upsides if u < -10])
                
                print(f"\n📈 VALUATION METRICS:")
                print(f"Average Upside/Downside: {avg_upside:+.1f}%")
                print(f"Strong Buy Signals (>10% upside): {positive_calls}")
                print(f"Strong Sell Signals (<-10% downside): {negative_calls}")
                print(f"Hold Range (-10% to +10%): {len(upsides) - positive_calls - negative_calls}")
            
            # Analyst alignment analysis
            aligned_count = 0
            divergent_count = 0
            for result in successful_results:
                alignment = result.get('consensus_alignment', '')
                if 'ALIGNED' in alignment:
                    aligned_count += 1
                elif 'DIVERGENT' in alignment:
                    divergent_count += 1
            
            if aligned_count + divergent_count > 0:
                alignment_rate = aligned_count / (aligned_count + divergent_count) * 100
                print(f"\n🎯 ANALYST CONSENSUS ALIGNMENT:")
                print(f"Aligned with Analysts: {aligned_count}/{aligned_count + divergent_count} ({alignment_rate:.1f}%)")
                print(f"Divergent from Analysts: {divergent_count}")
            
            # Sector breakdown
            sector_performance = {}
            for result in successful_results:
                sector = result.get('sector', 'Unknown')
                upside = result.get('primary_upside', 0)
                if sector not in sector_performance:
                    sector_performance[sector] = []
                sector_performance[sector].append(upside)
            
            print(f"\n🏭 SECTOR PERFORMANCE:")
            for sector, upsides in sector_performance.items():
                if len(upsides) >= 2:  # Only show sectors with multiple stocks
                    avg_sector_upside = sum(upsides) / len(upsides)
                    print(f"  {sector}: {avg_sector_upside:+.1f}% avg ({len(upsides)} stocks)")
            
            # Framework comparison
            hypergrowth_results = [r for r in successful_results if r.get('is_hypergrowth', False)]
            traditional_results = [r for r in successful_results if not r.get('is_hypergrowth', False)]
            
            if hypergrowth_results and traditional_results:
                hg_upsides = [r.get('primary_upside', 0) for r in hypergrowth_results if r.get('primary_upside') is not None]
                trad_upsides = [r.get('primary_upside', 0) for r in traditional_results if r.get('primary_upside') is not None]
                
                if hg_upsides and trad_upsides:
                    hg_avg = sum(hg_upsides) / len(hg_upsides)
                    trad_avg = sum(trad_upsides) / len(trad_upsides)
                    
                    print(f"\n⚖️  FRAMEWORK COMPARISON:")
                    print(f"Hypergrowth Framework Avg: {hg_avg:+.1f}% ({len(hg_upsides)} stocks)")
                    print(f"Traditional Framework Avg: {trad_avg:+.1f}% ({len(trad_upsides)} stocks)")
        
        return all_results
        """Test the hypergrowth framework on a specific stock."""
        try:
            print(f"\n{'='*60}")
            print(f"TESTING HYPERGROWTH FRAMEWORK: {ticker}")
            print(f"{'='*60}")
            
            stock = StockAnalysis(ticker)
            
            # Load data
            data_loaded = False
            try:
                stock.load_info_from_file()
                stock.load_analysis_from_file()
                stock.load_financials_from_file()
                stock.load_history_from_file()
                
                # Check if we actually got meaningful data
                info = stock.data.get('info', {})
                if info.get('currentPrice', 0) > 0:
                    data_loaded = True
                    print(f"✓ Loaded data from files")
                else:
                    print("Files exist but contain incomplete data")
            except Exception as e:
                print(f"Could not load from files: {e}")
            
            # Fetch fresh data if files are missing or incomplete
            if not data_loaded:
                print("Fetching fresh data from Yahoo Finance...")
                try:
                    stock.fetch_data()
                    stock.fetch_analysis()  
                    stock.fetch_financials()
                    stock.fetch_history()
                    print("✓ Fetched fresh data")
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    return False
            
            # Get basic info
            info = stock.data.get('info', {})
            current_price = info.get('currentPrice', 0)
            market_cap = info.get('marketCap', 0)
            
            print(f"\nCompany: {ticker}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Market Cap: ${market_cap:,.0f}")
            
            # Check if identified as hypergrowth
            is_hypergrowth = stock._identify_hypergrowth_company()
            print(f"Identified as Hypergrowth: {is_hypergrowth}")
            
            if is_hypergrowth:
                print(f"\n🚀 HYPERGROWTH ANALYSIS")
                print("-" * 40)
                
                # Run hypergrowth evaluation
                hypergrowth_value = stock.evaluate_hypergrowth_stock()
                
                if hypergrowth_value:
                    upside = ((hypergrowth_value - current_price) / current_price * 100)
                    print(f"Hypergrowth Fair Value: ${hypergrowth_value:.2f}")
                    print(f"Implied Upside: {upside:+.1f}%")
                    
                    # Compare with traditional DCF
                    print(f"\n📊 TRADITIONAL DCF COMPARISON")
                    print("-" * 40)
                    dcf_value = stock.evaluate_DCF()
                    if dcf_value:
                        dcf_upside = ((dcf_value - current_price) / current_price * 100)
                        print(f"Traditional DCF: ${dcf_value:.2f}")
                        print(f"DCF Upside: {dcf_upside:+.1f}%")
                        
                        print(f"\nDifference: Hypergrowth shows {upside - dcf_upside:+.1f}% more upside than DCF")
                    
                    # Get analyst consensus
                    print(f"\n📈 ANALYST CONSENSUS")
                    print("-" * 40)
                    analyst_data = stock.parse_analyst_recommendations()
                    if analyst_data:
                        print(f"Analyst Rating: {analyst_data.get('consensus', 'N/A')}")
                        print(f"Strong Buy: {analyst_data.get('strong_buy', 0)}")
                        print(f"Buy: {analyst_data.get('buy', 0)}")
                        print(f"Hold: {analyst_data.get('hold', 0)}")
                        print(f"Sell: {analyst_data.get('sell', 0)}")
                        print(f"Strong Sell: {analyst_data.get('strong_sell', 0)}")
                        
                        # Validate consensus
                        validation = stock.validate_against_analyst_consensus(
                            {'Hypergrowth': hypergrowth_value}, current_price, analyst_data
                        )
                        if validation:
                            print(f"\nConsensus Validation:")
                            print(f"  Model Alignment: {validation.get('alignment', 'Unknown')}")
                            print(f"  Model Sentiment: {validation.get('model_sentiment', 'Unknown')}")
                            print(f"  Analyst Sentiment: {validation.get('analyst_sentiment', 'Unknown')}")
                    
                else:
                    print("Could not calculate hypergrowth valuation")
            else:
                print(f"{ticker} not identified as hypergrowth - using traditional analysis")
                dcf_value = stock.evaluate_DCF()
                if dcf_value:
                    upside = ((dcf_value - current_price) / current_price * 100)
                    print(f"DCF Fair Value: ${dcf_value:.2f}")
                    print(f"Implied Upside: {upside:+.1f}%")
            
            return True
            
        except Exception as e:
            print(f"Error testing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the test based on command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "comprehensive" or command == "test":
            # Run comprehensive test suite
            sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            run_comprehensive_test_suite(sample_size)
            
        elif command == "hypergrowth":
            # Test specific hypergrowth stocks
            print("Testing hypergrowth framework on key stocks...")
            for ticker in hypergrowth_stocks:
                comprehensive_stock_analysis(ticker)
                print()
                
        else:
            # Analyze specific ticker
            ticker = command.upper()
            comprehensive_stock_analysis(ticker)
    else:
        # Interactive mode
        print("Magic Beans Stock Analysis")
        print("=" * 40)
        print("Available commands:")
        print("  python stock_analysis.py comprehensive [sample_size] - Test N random stocks (default 50)")
        print("  python stock_analysis.py hypergrowth - Test hypergrowth stocks")
        print("  python stock_analysis.py [TICKER] - Analyze specific stock")
        print()
        
        choice = input("Choose analysis type (comprehensive/hypergrowth/ticker): ").lower()
        
        if choice == "comprehensive":
            sample_size = input("Sample size (default 50): ")
            sample_size = int(sample_size) if sample_size.isdigit() else 50
            run_comprehensive_test_suite(sample_size)
        elif choice == "hypergrowth":
            for ticker in hypergrowth_stocks:
                comprehensive_stock_analysis(ticker)
                print()
        else:
            ticker = input("Enter ticker symbol: ").upper()
            if ticker:
                comprehensive_stock_analysis(ticker)