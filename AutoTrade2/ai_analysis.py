"""
AI Analysis Module
Advanced pattern recognition, probability models, and machine learning features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class AIAnalyzer:
    def __init__(self):
        """Initialize AI Analyzer"""
        self.pattern_history = []
        self.regime_history = []
        
    # ==================== Candlestick Pattern Recognition ====================
    
    def detect_doji(self, open_price: float, high: float, low: float, close: float, 
                    threshold: float = 0.1) -> Dict:
        """Detect Doji pattern"""
        body = abs(close - open_price)
        range_size = high - low
        
        if range_size == 0:
            return {'pattern': None, 'strength': 0}
        
        body_percentage = body / range_size
        
        if body_percentage < threshold:
            upper_shadow = high - max(open_price, close)
            lower_shadow = min(open_price, close) - low
            
            # Types of Doji
            if upper_shadow > body * 2 and lower_shadow < body:
                return {'pattern': 'dragonfly_doji', 'strength': 0.8, 'signal': 'bullish'}
            elif lower_shadow > body * 2 and upper_shadow < body:
                return {'pattern': 'gravestone_doji', 'strength': 0.8, 'signal': 'bearish'}
            else:
                return {'pattern': 'doji', 'strength': 0.6, 'signal': 'neutral'}
        
        return {'pattern': None, 'strength': 0}
    
    def detect_hammer(self, open_price: float, high: float, low: float, close: float) -> Dict:
        """Detect Hammer/Hanging Man pattern"""
        body = abs(close - open_price)
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Hammer: long lower shadow, small body, little to no upper shadow
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            if close > open_price:
                return {'pattern': 'hammer', 'strength': 0.85, 'signal': 'bullish'}
            else:
                return {'pattern': 'hanging_man', 'strength': 0.75, 'signal': 'bearish'}
        
        return {'pattern': None, 'strength': 0}
    
    def detect_shooting_star(self, open_price: float, high: float, low: float, close: float) -> Dict:
        """Detect Shooting Star/Inverted Hammer pattern"""
        body = abs(close - open_price)
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        # Shooting Star: long upper shadow, small body, little to no lower shadow
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            if close < open_price:
                return {'pattern': 'shooting_star', 'strength': 0.85, 'signal': 'bearish'}
            else:
                return {'pattern': 'inverted_hammer', 'strength': 0.75, 'signal': 'bullish'}
        
        return {'pattern': None, 'strength': 0}
    
    def detect_engulfing(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect Bullish/Bearish Engulfing pattern"""
        if index < 1:
            return {'pattern': None, 'strength': 0}
        
        prev_open = df['open'].iloc[index - 1]
        prev_close = df['close'].iloc[index - 1]
        curr_open = df['open'].iloc[index]
        curr_close = df['close'].iloc[index]
        
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        
        # Bullish Engulfing
        if (prev_close < prev_open and  # Previous bearish
            curr_close > curr_open and  # Current bullish
            curr_open < prev_close and  # Opens below previous close
            curr_close > prev_open and  # Closes above previous open
            curr_body > prev_body * 1.1):  # Larger body
            return {'pattern': 'bullish_engulfing', 'strength': 0.9, 'signal': 'bullish'}
        
        # Bearish Engulfing
        if (prev_close > prev_open and  # Previous bullish
            curr_close < curr_open and  # Current bearish
            curr_open > prev_close and  # Opens above previous close
            curr_close < prev_open and  # Closes below previous open
            curr_body > prev_body * 1.1):  # Larger body
            return {'pattern': 'bearish_engulfing', 'strength': 0.9, 'signal': 'bearish'}
        
        return {'pattern': None, 'strength': 0}
    
    def detect_morning_evening_star(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect Morning Star/Evening Star pattern"""
        if index < 2:
            return {'pattern': None, 'strength': 0}
        
        first_open = df['open'].iloc[index - 2]
        first_close = df['close'].iloc[index - 2]
        second_open = df['open'].iloc[index - 1]
        second_close = df['close'].iloc[index - 1]
        third_open = df['open'].iloc[index]
        third_close = df['close'].iloc[index]
        
        first_body = abs(first_close - first_open)
        second_body = abs(second_close - second_open)
        third_body = abs(third_close - third_open)
        
        # Morning Star (Bullish)
        if (first_close < first_open and  # First bearish
            second_body < first_body * 0.3 and  # Small second body (star)
            third_close > third_open and  # Third bullish
            third_close > (first_open + first_close) / 2):  # Closes above middle of first
            return {'pattern': 'morning_star', 'strength': 0.95, 'signal': 'bullish'}
        
        # Evening Star (Bearish)
        if (first_close > first_open and  # First bullish
            second_body < first_body * 0.3 and  # Small second body (star)
            third_close < third_open and  # Third bearish
            third_close < (first_open + first_close) / 2):  # Closes below middle of first
            return {'pattern': 'evening_star', 'strength': 0.95, 'signal': 'bearish'}
        
        return {'pattern': None, 'strength': 0}
    
    def detect_three_soldiers_crows(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect Three White Soldiers/Three Black Crows"""
        if index < 2:
            return {'pattern': None, 'strength': 0}
        
        closes = [df['close'].iloc[i] for i in range(index - 2, index + 1)]
        opens = [df['open'].iloc[i] for i in range(index - 2, index + 1)]
        
        # Three White Soldiers (Bullish)
        if all(closes[i] > opens[i] for i in range(3)) and \
           all(closes[i] > closes[i-1] for i in range(1, 3)):
            return {'pattern': 'three_white_soldiers', 'strength': 0.9, 'signal': 'bullish'}
        
        # Three Black Crows (Bearish)
        if all(closes[i] < opens[i] for i in range(3)) and \
           all(closes[i] < closes[i-1] for i in range(1, 3)):
            return {'pattern': 'three_black_crows', 'strength': 0.9, 'signal': 'bearish'}
        
        return {'pattern': None, 'strength': 0}
    
    def analyze_all_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze all candlestick patterns"""
        patterns_found = []
        
        for i in range(max(3, len(df) - 10), len(df)):  # Check last 10 candles
            # Single candle patterns
            doji = self.detect_doji(df['open'].iloc[i], df['high'].iloc[i], 
                                   df['low'].iloc[i], df['close'].iloc[i])
            if doji['pattern']:
                patterns_found.append({'index': i, **doji})
            
            hammer = self.detect_hammer(df['open'].iloc[i], df['high'].iloc[i],
                                       df['low'].iloc[i], df['close'].iloc[i])
            if hammer['pattern']:
                patterns_found.append({'index': i, **hammer})
            
            shooting = self.detect_shooting_star(df['open'].iloc[i], df['high'].iloc[i],
                                                df['low'].iloc[i], df['close'].iloc[i])
            if shooting['pattern']:
                patterns_found.append({'index': i, **shooting})
            
            # Multi-candle patterns
            if i >= 1:
                engulfing = self.detect_engulfing(df, i)
                if engulfing['pattern']:
                    patterns_found.append({'index': i, **engulfing})
            
            if i >= 2:
                star = self.detect_morning_evening_star(df, i)
                if star['pattern']:
                    patterns_found.append({'index': i, **star})
                
                soldiers_crows = self.detect_three_soldiers_crows(df, i)
                if soldiers_crows['pattern']:
                    patterns_found.append({'index': i, **soldiers_crows})
        
        return patterns_found
    
    # ==================== Market Regime Detection ====================
    
    def detect_market_regime(self, df: pd.DataFrame, period: int = 50) -> Dict:
        """
        Detect market regime using advanced analysis
        Returns: trending_up, trending_down, ranging, volatile, crisis
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['tick_volume']
        
        # Calculate metrics
        returns = close.pct_change()
        volatility = returns.rolling(window=period).std()
        avg_vol = volatility.mean()
        current_vol = volatility.iloc[-1]
        
        # Trend strength
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        trend_strength = (close.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        # ADX for trend strength
        atr_val = self._calculate_atr(high, low, close, 14)
        adx = self._calculate_adx(high, low, close, 14)
        
        # Volume analysis
        avg_volume = volume.rolling(window=period).mean().iloc[-1]
        recent_volume = volume.iloc[-10:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Price momentum
        momentum = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
        
        # Decision logic
        regime_score = {
            'trending_up': 0,
            'trending_down': 0,
            'ranging': 0,
            'volatile': 0,
            'crisis': 0
        }
        
        # Crisis detection (extreme volatility + volume spike)
        if current_vol > avg_vol * 2.5 and volume_ratio > 1.5:
            regime_score['crisis'] = 0.9
        
        # Volatile market
        elif current_vol > avg_vol * 1.5:
            regime_score['volatile'] = 0.8
        
        # Trending market
        elif adx > 25 and abs(trend_strength) > 0.02:
            if momentum > 0 and close.iloc[-1] > sma_20.iloc[-1]:
                regime_score['trending_up'] = 0.85
            elif momentum < 0 and close.iloc[-1] < sma_20.iloc[-1]:
                regime_score['trending_down'] = 0.85
            else:
                regime_score['ranging'] = 0.6
        
        # Ranging market
        else:
            regime_score['ranging'] = 0.75
        
        # Get dominant regime
        dominant_regime = max(regime_score, key=regime_score.get)
        confidence = regime_score[dominant_regime]
        
        return {
            'regime': dominant_regime,
            'confidence': confidence,
            'volatility_ratio': current_vol / avg_vol if avg_vol > 0 else 1,
            'trend_strength': abs(trend_strength),
            'adx': adx,
            'volume_ratio': volume_ratio,
            'momentum': momentum,
            'scores': regime_score
        }
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Calculate ADX"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(high, low, close, 1) * period
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.iloc[-1] if len(adx) > 0 else 0
    
    # ==================== Bayesian Probability ====================
    
    def bayesian_update(self, prior: float, likelihood: float, evidence: float) -> float:
        """
        Bayesian probability update
        P(A|B) = P(B|A) * P(A) / P(B)
        """
        if evidence == 0:
            return prior
        
        posterior = (likelihood * prior) / evidence
        return min(max(posterior, 0.0), 1.0)  # Clamp between 0 and 1
    
    def calculate_buy_sell_probability(self, analysis: Dict, patterns: List[Dict], 
                                      regime: Dict) -> Dict:
        """
        Calculate probability of successful buy/sell using Bayesian approach
        """
        # Prior probabilities (base assumption)
        buy_prior = 0.5
        sell_prior = 0.5
        
        # Evidence from technical indicators
        buy_evidence = 0
        sell_evidence = 0
        total_factors = 0
        
        if 'rsi' in analysis:
            rsi = analysis['rsi']
            if rsi < 30:
                buy_evidence += 0.8
            elif rsi < 40:
                buy_evidence += 0.5
            elif rsi > 70:
                sell_evidence += 0.8
            elif rsi > 60:
                sell_evidence += 0.5
            total_factors += 1
        
        if 'macd_histogram' in analysis:
            if analysis['macd_histogram'] > 0:
                buy_evidence += 0.6
            else:
                sell_evidence += 0.6
            total_factors += 1
        
        if 'sma_20' in analysis and 'sma_50' in analysis:
            if analysis['sma_20'] > analysis['sma_50']:
                buy_evidence += 0.7
            else:
                sell_evidence += 0.7
            total_factors += 1
        
        if all(k in analysis for k in ['current_price', 'bb_upper', 'bb_lower']):
            bb_position = (analysis['current_price'] - analysis['bb_lower']) / \
                         (analysis['bb_upper'] - analysis['bb_lower'])
            if bb_position < 0.2:
                buy_evidence += 0.75
            elif bb_position > 0.8:
                sell_evidence += 0.75
            total_factors += 1
        
        # ADX trend strength
        if 'adx' in analysis:
            adx_multiplier = min(analysis['adx'] / 25, 1.5)  # Strong trend boost
            if analysis.get('plus_di', 0) > analysis.get('minus_di', 0):
                buy_evidence *= adx_multiplier
            else:
                sell_evidence *= adx_multiplier
        
        # Pattern analysis
        pattern_buy_score = 0
        pattern_sell_score = 0
        
        for pattern in patterns:
            if pattern.get('signal') == 'bullish':
                pattern_buy_score += pattern.get('strength', 0)
            elif pattern.get('signal') == 'bearish':
                pattern_sell_score += pattern.get('strength', 0)
        
        if pattern_buy_score > 0:
            buy_evidence += pattern_buy_score
            total_factors += 1
        if pattern_sell_score > 0:
            sell_evidence += pattern_sell_score
            total_factors += 1
        
        # Market regime adjustment
        if regime['regime'] == 'trending_up':
            buy_evidence *= 1.3
        elif regime['regime'] == 'trending_down':
            sell_evidence *= 1.3
        elif regime['regime'] in ['volatile', 'crisis']:
            # Reduce confidence in volatile markets
            buy_evidence *= 0.7
            sell_evidence *= 0.7
        
        # Normalize evidence
        if total_factors > 0:
            buy_likelihood = buy_evidence / total_factors
            sell_likelihood = sell_evidence / total_factors
        else:
            buy_likelihood = 0.5
            sell_likelihood = 0.5
        
        # Calculate posterior probabilities
        total_evidence = buy_likelihood + sell_likelihood
        if total_evidence > 0:
            buy_probability = self.bayesian_update(buy_prior, buy_likelihood, total_evidence)
            sell_probability = self.bayesian_update(sell_prior, sell_likelihood, total_evidence)
        else:
            buy_probability = 0.5
            sell_probability = 0.5
        
        # Normalize to sum to 1
        total = buy_probability + sell_probability
        if total > 0:
            buy_probability /= total
            sell_probability /= total
        
        return {
            'buy_probability': buy_probability,
            'sell_probability': sell_probability,
            'confidence': abs(buy_probability - sell_probability),
            'recommendation': 'BUY' if buy_probability > sell_probability else 'SELL',
            'strength': abs(buy_probability - sell_probability) * 100
        }
    
    # ==================== Momentum Quality Index ====================
    
    def momentum_quality_index(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """
        Calculate Momentum Quality Index
        Measures the quality and sustainability of momentum
        """
        close = df['close']
        volume = df['tick_volume']
        high = df['high']
        low = df['low']
        
        # Price momentum
        price_change = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]
        
        # Volume momentum
        avg_volume = volume.rolling(window=period).mean()
        recent_volume = volume.iloc[-5:].mean()
        volume_quality = recent_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        
        # Trend consistency
        returns = close.pct_change()
        positive_days = (returns > 0).rolling(window=period).sum().iloc[-1]
        consistency = positive_days / period if price_change > 0 else (period - positive_days) / period
        
        # Volatility factor
        volatility = returns.rolling(window=period).std().iloc[-1]
        volatility_score = 1 - min(volatility * 10, 1)  # Lower volatility = higher quality
        
        # Calculate composite score
        momentum_score = (
            abs(price_change) * 0.3 +
            volume_quality * 0.2 +
            consistency * 0.3 +
            volatility_score * 0.2
        ) * 100
        
        return {
            'momentum_quality': min(momentum_score, 100),
            'price_momentum': price_change * 100,
            'volume_quality': volume_quality,
            'trend_consistency': consistency,
            'volatility_score': volatility_score
        }
    
    # ==================== Volume Profile Analysis ====================
    
    def volume_profile_analysis(self, df: pd.DataFrame, bins: int = 20) -> Dict:
        """
        Analyze volume distribution across price levels
        """
        close = df['close']
        volume = df['tick_volume']
        
        # Calculate price range
        price_min = close.min()
        price_max = close.max()
        
        # Create price bins
        price_bins = np.linspace(price_min, price_max, bins + 1)
        volume_profile = np.zeros(bins)
        
        # Accumulate volume in each price bin
        for i in range(len(df)):
            price = close.iloc[i]
            vol = volume.iloc[i]
            bin_index = min(int((price - price_min) / (price_max - price_min) * bins), bins - 1)
            volume_profile[bin_index] += vol
        
        # Find Point of Control (POC) - price level with highest volume
        poc_index = np.argmax(volume_profile)
        poc_price = price_bins[poc_index] + (price_bins[poc_index + 1] - price_bins[poc_index]) / 2
        
        # Find Value Area (70% of volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.7
        
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            value_area_indices.append(idx)
            cumulative_volume += volume_profile[idx]
            if cumulative_volume >= target_volume:
                break
        
        value_area_low = price_bins[min(value_area_indices)]
        value_area_high = price_bins[max(value_area_indices) + 1]
        
        current_price = close.iloc[-1]
        
        return {
            'poc_price': poc_price,
            'value_area_low': value_area_low,
            'value_area_high': value_area_high,
            'current_vs_poc': (current_price - poc_price) / poc_price * 100,
            'in_value_area': value_area_low <= current_price <= value_area_high,
            'volume_distribution': 'balanced' if abs(current_price - poc_price) / poc_price < 0.01 else 'skewed'
        }
    
    # ==================== Complete AI Analysis ====================
    
    def analyze_complete(self, df: pd.DataFrame, technical_analysis: Dict) -> Dict:
        """
        Perform complete AI analysis
        """
        # Pattern recognition
        patterns = self.analyze_all_patterns(df)
        
        # Market regime
        regime = self.detect_market_regime(df)
        
        # Probabilities
        probabilities = self.calculate_buy_sell_probability(technical_analysis, patterns, regime)
        
        # Momentum quality
        momentum_quality = self.momentum_quality_index(df)
        
        # Volume profile
        volume_profile = self.volume_profile_analysis(df)
        
        return {
            'timestamp': datetime.now(),
            'patterns': patterns,
            'regime': regime,
            'probabilities': probabilities,
            'momentum_quality': momentum_quality,
            'volume_profile': volume_profile,
            'pattern_count': len(patterns),
            'dominant_pattern': patterns[-1] if patterns else None
        }
