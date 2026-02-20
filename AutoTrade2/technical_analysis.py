"""
Technical Analysis Module
Advanced indicators and calculations for trading decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import MetaTrader5 as mt5
from datetime import datetime, timedelta

class TechnicalAnalyzer:
    def __init__(self):
        """Initialize Technical Analyzer"""
        self.cache = {}
        
    def get_ohlcv_data(self, symbol: str, timeframe: int, bars: int = 500) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data from MT5
        
        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            return None
    
    # ==================== Moving Averages ====================
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def wma(self, data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def kama(self, data: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average"""
        try:
            change = abs(data - data.shift(period))
            volatility = data.diff().abs().rolling(window=period).sum()
            
            # ป้องกัน division by zero
            volatility_safe = volatility.replace(0, np.nan)
            er = change / volatility_safe  # Efficiency Ratio
            er = er.fillna(0)  # flat market = 0 efficiency
            
            fast_sc = 2 / (fast + 1)
            slow_sc = 2 / (slow + 1)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            kama = pd.Series(index=data.index, dtype=float)
            kama.iloc[period - 1] = data.iloc[period - 1]
            
            for i in range(period, len(data)):
                kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i - 1])
            
            return kama
        except Exception as e:
            # Return simple moving average as fallback
            return self.sma(data, period)
    
    # ==================== Momentum Indicators ====================
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # ป้องกัน division by zero
            loss_safe = loss.replace(0, np.nan)
            rs = gain / loss_safe
            
            # จัดการกับ inf และ nan
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.replace([np.inf, -np.inf], 100)  # ถ้า loss = 0 แปลว่า RSI = 100
            rsi = rsi.fillna(50)  # neutral value
            
            return rsi
        except Exception as e:
            # Return neutral RSI
            return pd.Series([50] * len(data), index=data.index)
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            
            # ป้องกัน division by zero เมื่อ market flat
            range_hl = highest_high - lowest_low
            range_hl = range_hl.replace(0, np.nan)  # แทนที่ 0 เป็น NaN
            
            k = 100 * (close - lowest_low) / range_hl
            k = k.fillna(50)  # neutral value สำหรับ flat market
            k = k.rolling(window=smooth_k).mean()
            d = k.rolling(window=smooth_d).mean()
            
            return {'k': k, 'd': d}
        except Exception as e:
            # Return neutral values
            neutral = pd.Series([50] * len(close), index=close.index)
            return {'k': neutral, 'd': neutral}
    
    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, 
            volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        try:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            delta = typical_price.diff()
            positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
            negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()
            
            # ป้องกัน division by zero
            negative_flow = negative_flow.replace(0, np.nan)
            money_ratio = positive_flow / negative_flow
            money_ratio = money_ratio.fillna(1)  # neutral ratio
            
            mfi = 100 - (100 / (1 + money_ratio))
            return mfi
        except Exception as e:
            # Return neutral MFI
            return pd.Series([50] * len(close), index=close.index)
    
    # ==================== Volatility Indicators ====================
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        try:
            middle = self.sma(data, period)
            std_dev = data.rolling(window=period).std()
            
            # ป้องกัน division by zero สำหรับ bandwidth
            middle_safe = middle.replace(0, np.nan)
            bandwidth = (std_dev * std * 2) / middle_safe * 100
            bandwidth = bandwidth.fillna(0)
            
            return {
                'upper': middle + (std_dev * std),
                'middle': middle,
                'lower': middle - (std_dev * std),
                'bandwidth': bandwidth
            }
        except Exception as e:
            # Return safe defaults
            return {
                'upper': data * 1.02,
                'middle': data,
                'lower': data * 0.98,
                'bandwidth': pd.Series([2.0] * len(data), index=data.index)
            }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """Keltner Channels"""
        middle = self.ema(close, period)
        atr_val = self.atr(high, low, close, period)
        
        return {
            'upper': middle + (multiplier * atr_val),
            'middle': middle,
            'lower': middle - (multiplier * atr_val)
        }
    
    # ==================== Trend Indicators ====================
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        try:
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = self.atr(high, low, close, 1) * period
            
            # ป้องกัน division by zero
            tr_safe = tr.replace(0, np.nan)
            plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr_safe)
            minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr_safe)
            plus_di = plus_di.fillna(25)
            minus_di = minus_di.fillna(25)
            
            # ป้องกัน division by zero สำหรับ DX
            di_sum = plus_di + minus_di
            di_sum_safe = di_sum.replace(0, np.nan)
            dx = 100 * abs(plus_di - minus_di) / di_sum_safe
            dx = dx.fillna(0)
            
            adx = dx.rolling(window=period).mean()
            
            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di
            }
        except Exception as e:
            # Return neutral values
            neutral = pd.Series([25] * len(close), index=close.index)
            return {
                'adx': neutral,
                'plus_di': neutral,
                'minus_di': neutral
            }
    
    def ichimoku(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line): 9-period
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): 26-period
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    # ==================== Volume Indicators ====================
    
    def vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        
        return obv
    
    # ==================== Support/Resistance ====================
    
    def find_support_resistance(self, high: pd.Series, low: pd.Series, 
                               close: pd.Series, window: int = 20) -> Dict[str, List[float]]:
        """Find Support and Resistance levels using pivot points"""
        supports = []
        resistances = []
        
        for i in range(window, len(close) - window):
            # Check if current point is a local minimum (support)
            if low.iloc[i] == low.iloc[i - window:i + window + 1].min():
                supports.append(low.iloc[i])
            
            # Check if current point is a local maximum (resistance)
            if high.iloc[i] == high.iloc[i - window:i + window + 1].max():
                resistances.append(high.iloc[i])
        
        # Cluster similar levels
        supports = self._cluster_levels(supports, close.iloc[-1] * 0.001)
        resistances = self._cluster_levels(resistances, close.iloc[-1] * 0.001)
        
        return {
            'supports': sorted(supports)[-5:],  # Top 5 support levels
            'resistances': sorted(resistances)[:5]  # Top 5 resistance levels
        }
    
    def _cluster_levels(self, levels: List[float], threshold: float) -> List[float]:
        """Cluster similar price levels"""
        if not levels:
            return []
        
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - np.mean(current_cluster)) <= threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered
    
    def pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Pivot Points (Standard)"""
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
    
    # ==================== Fibonacci ====================
    
    def fibonacci_retracement(self, high: float, low: float, direction: str = 'up') -> Dict[str, float]:
        """Calculate Fibonacci Retracement levels"""
        diff = high - low
        
        if direction == 'up':
            return {
                '0.0': high,
                '0.236': high - 0.236 * diff,
                '0.382': high - 0.382 * diff,
                '0.5': high - 0.5 * diff,
                '0.618': high - 0.618 * diff,
                '0.786': high - 0.786 * diff,
                '1.0': low
            }
        else:
            return {
                '0.0': low,
                '0.236': low + 0.236 * diff,
                '0.382': low + 0.382 * diff,
                '0.5': low + 0.5 * diff,
                '0.618': low + 0.618 * diff,
                '0.786': low + 0.786 * diff,
                '1.0': high
            }
    
    # ==================== Advanced Analysis ====================
    
    def hurst_exponent(self, data: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate Hurst Exponent (Fractal Dimension)
        Returns: 0-0.5 = mean reverting, 0.5 = random walk, 0.5-1 = trending
        """
        try:
            if len(data) < max_lag + 10:
                return 0.5  # Not enough data, assume random walk
            
            lags = range(2, max_lag)
            tau = []
            
            for lag in lags:
                pp = np.subtract(data[lag:].values, data[:-lag].values)
                std_val = np.std(pp)
                
                # เช็คว่า std เป็น 0 หรือไม่ (ราคาไม่เคลื่อนไหว)
                if std_val <= 0:
                    continue
                    
                tau.append(std_val)
            
            # ต้องมีข้อมูลพอสำหรับ regression
            if len(tau) < 3:
                return 0.5
            
            # เอา lags ที่ตรงกับ tau ที่เหลือ
            valid_lags = list(lags)[:len(tau)]
            
            # แปลงเป็น log (ต้องมั่นใจว่าไม่มี 0)
            log_lags = np.log(valid_lags)
            log_tau = np.log(tau)
            
            # เช็คว่ามี inf หรือ nan หรือไม่
            if np.any(np.isinf(log_lags)) or np.any(np.isnan(log_lags)) or \
               np.any(np.isinf(log_tau)) or np.any(np.isnan(log_tau)):
                return 0.5
            
            # คำนวณ Hurst Exponent
            poly = np.polyfit(log_lags, log_tau, 1)
            hurst = poly[0]
            
            # จำกัดค่าให้อยู่ในช่วง 0-1
            hurst = max(0.0, min(1.0, hurst))
            
            return hurst
            
        except Exception as e:
            # ถ้าเกิด error ใดๆ ให้ return 0.5 (random walk)
            return 0.5
    
    def linear_regression(self, data: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """
        Linear Regression with Standard Error
        """
        try:
            lr = pd.Series(index=data.index, dtype=float)
            stderr = pd.Series(index=data.index, dtype=float)
            
            for i in range(period - 1, len(data)):
                y = data.iloc[i - period + 1:i + 1].values
                x = np.arange(len(y))
                
                # เช็คว่ามีข้อมูลพอ
                if len(y) < 2:
                    continue
                
                # Linear regression
                coeffs = np.polyfit(x, y, 1)
                lr.iloc[i] = coeffs[0] * (period - 1) + coeffs[1]
                
                # Standard error
                y_pred = np.polyval(coeffs, x)
                residuals = y - y_pred
                
                # ป้องกัน division by zero
                if len(y) > 0:
                    stderr.iloc[i] = np.sqrt(np.sum(residuals ** 2) / len(y))
                else:
                    stderr.iloc[i] = 0
            
            return {
                'lr': lr,
                'stderr': stderr,
                'upper': lr + 2 * stderr,
                'lower': lr - 2 * stderr
            }
            
        except Exception as e:
            # Return empty series if error
            empty_series = pd.Series(index=data.index, dtype=float)
            return {
                'lr': empty_series,
                'stderr': empty_series,
                'upper': empty_series,
                'lower': empty_series
            }

    
    def market_regime(self, data: pd.Series, period: int = 20) -> str:
        """Detect market regime: trending, ranging, or volatile"""
        try:
            returns = data.pct_change()
            volatility = returns.rolling(window=period).std()
            
            # ป้องกัน division by zero
            start_price = data.iloc[-period]
            if start_price == 0:
                return "ranging"
            
            trend_strength = abs(data.iloc[-1] - start_price) / start_price
            
            avg_vol = volatility.mean()
            current_vol = volatility.iloc[-1]
            
            # เช็ค NaN
            if pd.isna(current_vol) or pd.isna(avg_vol):
                return "ranging"
            
            if current_vol > avg_vol * 1.5:
                return "volatile"
            elif trend_strength > 0.02:
                return "trending"
            else:
                return "ranging"
        except Exception as e:
            return "ranging"
    
    def correlation_analysis(self, price: pd.Series, volume: pd.Series, period: int = 20) -> float:
        """Analyze correlation between price and volume"""
        price_change = price.pct_change()
        volume_change = volume.pct_change()
        
        correlation = price_change.rolling(window=period).corr(volume_change)
        return correlation.iloc[-1]
    
    def growth_rate(self, data: pd.Series, period: int = 20) -> float:
        """Calculate Compound Annual Growth Rate (CAGR) style growth"""
        if len(data) < period:
            return 0.0
        
        start_val = data.iloc[-period]
        end_val = data.iloc[-1]
        
        if start_val <= 0:
            return 0.0
        
        growth = (end_val / start_val) ** (1 / period) - 1
        return growth * 100
    
    # ==================== Complete Analysis ====================
    
    def analyze_complete(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M15, 
                        bars: int = 500) -> Dict:
        """
        Perform complete technical analysis
        
        Returns:
            Dictionary with all analysis results
        """
        df = self.get_ohlcv_data(symbol, timeframe, bars)
        if df is None:
            return {'error': 'Failed to fetch data'}
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['tick_volume']
        
        # Moving Averages
        sma_20 = self.sma(close, 20)
        sma_50 = self.sma(close, 50)
        sma_200 = self.sma(close, 200)
        ema_12 = self.ema(close, 12)
        ema_26 = self.ema(close, 26)
        kama_val = self.kama(close)
        
        # Momentum
        rsi_val = self.rsi(close)
        macd_dict = self.macd(close)
        stoch_dict = self.stochastic(high, low, close)
        mfi_val = self.mfi(high, low, close, volume)
        
        # Volatility
        bb_dict = self.bollinger_bands(close)
        atr_val = self.atr(high, low, close)
        
        # Trend
        adx_dict = self.adx(high, low, close)
        ichimoku_dict = self.ichimoku(high, low, close)
        
        # Volume
        vwap_val = self.vwap(high, low, close, volume)
        obv_val = self.obv(close, volume)
        
        # Support/Resistance
        sr_levels = self.find_support_resistance(high, low, close)
        pivot = self.pivot_points(high.iloc[-1], low.iloc[-1], close.iloc[-1])
        
        # Advanced
        hurst = self.hurst_exponent(close)
        lr_dict = self.linear_regression(close)
        regime = self.market_regime(close)
        correlation = self.correlation_analysis(close, volume)
        growth = self.growth_rate(close)
        
        # Helper function สำหรับดึงค่าอย่างปลอดภัย
        def safe_get_value(series, default=0.0):
            try:
                if isinstance(series, pd.Series) and len(series) > 0:
                    val = series.iloc[-1]
                    # เช็ค NaN, inf
                    if pd.isna(val) or np.isinf(val):
                        return default
                    return val
                return default
            except (IndexError, KeyError):
                return default
        
        # Current values
        current_price = close.iloc[-1]
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'timestamp': datetime.now(),
            
            # Moving Averages
            'sma_20': safe_get_value(sma_20),
            'sma_50': safe_get_value(sma_50),
            'sma_200': safe_get_value(sma_200),
            'ema_12': safe_get_value(ema_12),
            'ema_26': safe_get_value(ema_26),
            'kama': safe_get_value(kama_val),
            
            # Momentum
            'rsi': safe_get_value(rsi_val, 50.0),
            'macd': safe_get_value(macd_dict['macd']),
            'macd_signal': safe_get_value(macd_dict['signal']),
            'macd_histogram': safe_get_value(macd_dict['histogram']),
            'stoch_k': safe_get_value(stoch_dict['k'], 50.0),
            'stoch_d': safe_get_value(stoch_dict['d'], 50.0),
            'mfi': safe_get_value(mfi_val, 50.0),
            
            # Volatility
            'bb_upper': safe_get_value(bb_dict['upper'], current_price * 1.02),
            'bb_middle': safe_get_value(bb_dict['middle'], current_price),
            'bb_lower': safe_get_value(bb_dict['lower'], current_price * 0.98),
            'bb_bandwidth': safe_get_value(bb_dict['bandwidth'], 0.02),
            'atr': safe_get_value(atr_val, current_price * 0.01),
            
            # Trend
            'adx': safe_get_value(adx_dict['adx'], 25.0),
            'plus_di': safe_get_value(adx_dict['plus_di'], 25.0),
            'minus_di': safe_get_value(adx_dict['minus_di'], 25.0),
            'ichimoku_tenkan': safe_get_value(ichimoku_dict['tenkan_sen'], current_price),
            'ichimoku_kijun': safe_get_value(ichimoku_dict['kijun_sen'], current_price),
            
            # Volume
            'vwap': safe_get_value(vwap_val, current_price),
            'obv': safe_get_value(obv_val, 0.0),
            
            # Support/Resistance
            'supports': sr_levels['supports'],
            'resistances': sr_levels['resistances'],
            'pivot_points': pivot,
            
            # Advanced
            'hurst_exponent': hurst if not (np.isnan(hurst) or np.isinf(hurst)) else 0.5,
            'lr_value': safe_get_value(lr_dict['lr'], current_price),
            'lr_stderr': safe_get_value(lr_dict['stderr'], 0.0),
            'market_regime': regime if regime else 'ranging',
            'price_volume_correlation': correlation if not (np.isnan(correlation) or np.isinf(correlation)) else 0.0,
            'growth_rate': growth if not (np.isnan(growth) or np.isinf(growth)) else 0.0,
            
            # Raw data for further analysis
            'dataframe': df
        }
