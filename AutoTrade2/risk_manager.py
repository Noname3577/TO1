"""
Risk Management Module
Advanced position sizing, stop loss/take profit, and risk calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import MetaTrader5 as mt5

class RiskManager:
    def __init__(self, account_balance: float = 10000, risk_percent: float = 2.0):
        """
        Initialize Risk Manager with Advanced 2026 Protection Systems
        
        Args:
            account_balance: Account balance in account currency
            risk_percent: Percentage of account to risk per trade (default 2%)
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance  # เก็บยอดเริ่มต้น
        self.risk_percent = risk_percent
        self.max_risk_percent = 5.0  # Maximum risk allowed
        self.min_risk_percent = 0.5  # Minimum risk
        self.trade_history = []
        
        # 🛡️ Advanced Risk Protection 2026
        # 1. Circuit Breakers
        self.daily_loss_limit_percent = 5.0  # หยุดเทรดถ้าขาดทุนเกิน 5%/วัน
        self.max_drawdown_percent = 15.0  # หยุดเทรดถ้า drawdown เกิน 15%
        self.circuit_breaker_active = False
        self.circuit_breaker_reset_time = None
        
        # 2. Drawdown Tracking
        self.peak_balance = account_balance
        self.current_drawdown_percent = 0.0
        
        # 3. Daily Loss Tracking
        self.daily_start_balance = account_balance
        self.daily_loss = 0.0
        self.last_reset_date = None
        
        # 4. Consecutive Loss Protection
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.reduce_risk_after_losses = True
        
        # 5. Volatility Protection
        self.max_volatility_multiplier = 3.0  # หยุดเทรดถ้า ATR สูงเกิน 3x ปกติ
        self.avg_atr_history = []
        
        # 6. Position Limits
        self.max_daily_trades = 20  # จำกัดจำนวนเทรดต่อวัน
        self.daily_trade_count = 0
        
        # 7. Stop Loss Per Trade (USD)
        self.max_loss_per_trade = 0.0  # จำกัดขาดทุนต่อออเดอร์ ($), 0 = ไม่จำกัด
        
    def update_balance(self, new_balance: float):
        """Update account balance"""
        self.account_balance = new_balance
    
    # ==================== Position Sizing ====================
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, risk_amount: Optional[float] = None) -> Dict:
        """
        Calculate optimal position size based on risk
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_amount: Fixed risk amount (if None, uses risk_percent)
            
        Returns:
            Dictionary with position sizing details
        """
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'error': 'Symbol not found'}
        
        # Calculate risk amount
        if risk_amount is None:
            risk_amount = self.account_balance * (self.risk_percent / 100)
        
        # Calculate pip value
        point = symbol_info.point
        pip_size = point * 10 if symbol_info.digits in [3, 5] else point
        
        # Calculate distance to stop loss in pips
        sl_distance_price = abs(entry_price - stop_loss)
        sl_distance_pips = sl_distance_price / pip_size
        
        if sl_distance_pips == 0:
            return {'error': 'Invalid stop loss distance'}
        
        # Calculate pip value per lot
        contract_size = symbol_info.trade_contract_size
        pip_value_per_lot = pip_size * contract_size
        
        # Calculate position size
        lot_size = risk_amount / (sl_distance_pips * pip_value_per_lot)
        
        # Round to valid lot size
        volume_step = symbol_info.volume_step
        lot_size = round(lot_size / volume_step) * volume_step
        
        # ⚠️ จำกัด lot size สูงสุดเพื่อป้องกัน margin blow-up
        # เพิ่มจาก original (เฉพาะ volume_max) 
        max_safe_lots = min(1.0, symbol_info.volume_max)  # จำกัดไม่เกิน 1.0 lot
        if lot_size > max_safe_lots:
            print(f"⚠️  Position size จำกัดจาก {lot_size:.2f} → {max_safe_lots:.2f} lots")
            lot_size = max_safe_lots
        
        # Apply limits
        lot_size = max(symbol_info.volume_min, lot_size)
        
        # Calculate actual risk
        actual_risk = lot_size * sl_distance_pips * pip_value_per_lot
        actual_risk_percent = (actual_risk / self.account_balance) * 100
        
        return {
            'lot_size': lot_size,
            'risk_amount': actual_risk,
            'risk_percent': actual_risk_percent,
            'sl_distance_pips': sl_distance_pips,
            'pip_value': pip_value_per_lot,
            'valid': symbol_info.volume_min <= lot_size <= symbol_info.volume_max
        }
    
    def calculate_lot_size_by_percent(self, symbol: str, entry_price: float,
                                     stop_loss: float, risk_percent: float) -> float:
        """Calculate lot size for a specific risk percentage"""
        risk_amount = self.account_balance * (risk_percent / 100)
        result = self.calculate_position_size(symbol, entry_price, stop_loss, risk_amount)
        return result.get('lot_size', 0.01)
    
    # ==================== Dynamic Stop Loss & Take Profit ====================
    
    def calculate_dynamic_sl_tp(self, symbol: str, order_type: str, 
                                entry_price: float, atr: float,
                                atr_multiplier_sl: float = 2.0,
                                atr_multiplier_tp: float = 3.0,
                                rr_ratio: float = 2.0) -> Dict:
        """
        Calculate dynamic Stop Loss and Take Profit based on ATR
        
        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            entry_price: Entry price
            atr: Average True Range value
            atr_multiplier_sl: ATR multiplier for stop loss
            atr_multiplier_tp: ATR multiplier for take profit
            rr_ratio: Risk/Reward ratio
            
        Returns:
            Dictionary with SL/TP levels
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'error': 'Symbol not found'}
        
        # Calculate SL/TP distances
        sl_distance = atr * atr_multiplier_sl
        tp_distance = sl_distance * rr_ratio  # Use RR ratio
        
        # Alternative: use ATR multiplier
        if atr_multiplier_tp:
            tp_distance = atr * atr_multiplier_tp
        
        if order_type.lower() == 'buy':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # sell
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # Calculate pips
        point = symbol_info.point
        pip_size = point * 10 if symbol_info.digits in [3, 5] else point
        
        sl_pips = abs(entry_price - stop_loss) / pip_size
        tp_pips = abs(entry_price - take_profit) / pip_size
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance,
            'risk_reward_ratio': tp_pips / sl_pips if sl_pips > 0 else 0
        }
    
    def trailing_stop_loss(self, symbol: str, order_type: str, entry_price: float,
                          current_price: float, initial_sl: float, 
                          trail_distance_pips: float) -> float:
        """
        Calculate trailing stop loss
        
        Args:
            symbol: Trading symbol
            order_type: 'buy' or 'sell'
            entry_price: Entry price
            current_price: Current market price
            initial_sl: Initial stop loss
            trail_distance_pips: Trailing distance in pips
            
        Returns:
            New stop loss level
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return initial_sl
        
        point = symbol_info.point
        pip_size = point * 10 if symbol_info.digits in [3, 5] else point
        trail_distance = trail_distance_pips * pip_size
        
        if order_type.lower() == 'buy':
            # For buy orders, move SL up if price moves up
            profit = current_price - entry_price
            if profit > trail_distance:
                new_sl = current_price - trail_distance
                return max(new_sl, initial_sl)  # Only move up, never down
        else:  # sell
            # For sell orders, move SL down if price moves down
            profit = entry_price - current_price
            if profit > trail_distance:
                new_sl = current_price + trail_distance
                return min(new_sl, initial_sl)  # Only move down, never up
        
        return initial_sl
    
    # ==================== Risk/Reward Analysis ====================
    
    def calculate_risk_reward(self, entry_price: float, stop_loss: float,
                             take_profit: float) -> Dict:
        """Calculate Risk/Reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return {'error': 'Invalid stop loss'}
        
        rr_ratio = reward / risk
        
        return {
            'risk': risk,
            'reward': reward,
            'risk_reward_ratio': rr_ratio,
            'rating': self._rate_risk_reward(rr_ratio)
        }
    
    def _rate_risk_reward(self, rr_ratio: float) -> str:
        """Rate the risk/reward ratio"""
        if rr_ratio >= 3.0:
            return 'Excellent'
        elif rr_ratio >= 2.0:
            return 'Good'
        elif rr_ratio >= 1.5:
            return 'Acceptable'
        elif rr_ratio >= 1.0:
            return 'Poor'
        else:
            return 'Unacceptable'
    
    # ==================== Win Rate & Expected Value ====================
    
    def estimate_win_rate(self, analysis: Dict, ai_analysis: Dict) -> Dict:
        """
        Estimate win rate based on analysis
        
        Args:
            analysis: Technical analysis results
            ai_analysis: AI analysis results
            
        Returns:
            Estimated win rate and confidence
        """
        win_rate = 50.0  # Base win rate
        confidence_factors = []
        
        # Factor 1: Probability from AI
        if 'probabilities' in ai_analysis:
            prob = ai_analysis['probabilities']
            max_prob = max(prob.get('buy_probability', 0.5), prob.get('sell_probability', 0.5))
            win_rate += (max_prob - 0.5) * 40  # Add up to 20%
            confidence_factors.append(prob.get('confidence', 0))
        
        # Factor 2: Market regime
        if 'regime' in ai_analysis:
            regime = ai_analysis['regime']
            if regime['regime'] in ['trending_up', 'trending_down']:
                win_rate += regime['confidence'] * 10
                confidence_factors.append(regime['confidence'])
            elif regime['regime'] in ['volatile', 'crisis']:
                win_rate -= 10  # Reduce confidence in volatile markets
        
        # Factor 3: Pattern strength
        if 'patterns' in ai_analysis and ai_analysis['patterns']:
            avg_strength = np.mean([p.get('strength', 0) for p in ai_analysis['patterns']])
            win_rate += avg_strength * 10
            confidence_factors.append(avg_strength)
        
        # Factor 4: Momentum quality
        if 'momentum_quality' in ai_analysis:
            mq = ai_analysis['momentum_quality']
            win_rate += (mq.get('momentum_quality', 50) - 50) * 0.2
            confidence_factors.append(mq.get('trend_consistency', 0.5))
        
        # Factor 5: ADX (trend strength)
        if 'adx' in analysis and analysis['adx'] > 25:
            win_rate += min((analysis['adx'] - 25) * 0.5, 10)
            confidence_factors.append(min(analysis['adx'] / 50, 1.0))
        
        # Clamp win rate
        win_rate = max(30.0, min(win_rate, 85.0))
        
        # Calculate confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5
        
        return {
            'estimated_win_rate': win_rate,
            'confidence': confidence,
            'rating': self._rate_win_rate(win_rate)
        }
    
    def _rate_win_rate(self, win_rate: float) -> str:
        """Rate the win rate"""
        if win_rate >= 70:
            return 'Excellent'
        elif win_rate >= 60:
            return 'Good'
        elif win_rate >= 50:
            return 'Fair'
        else:
            return 'Poor'
    
    def calculate_expected_value(self, win_rate: float, risk_reward_ratio: float,
                                risk_amount: float) -> Dict:
        """
        Calculate Expected Value (EV) of a trade
        
        EV = (Win Rate × Reward) - (Loss Rate × Risk)
        """
        win_rate_decimal = win_rate / 100
        loss_rate = 1 - win_rate_decimal
        
        reward = risk_amount * risk_reward_ratio
        
        ev = (win_rate_decimal * reward) - (loss_rate * risk_amount)
        ev_percent = (ev / risk_amount) * 100 if risk_amount > 0 else 0
        
        return {
            'expected_value': ev,
            'ev_percent': ev_percent,
            'expected_return': ev_percent,
            'profitable': ev > 0,
            'rating': 'Positive' if ev > 0 else 'Negative'
        }
    
    # ==================== Trade Quality Score ====================
    
    def calculate_trade_quality_score(self, analysis: Dict, ai_analysis: Dict,
                                     risk_reward_ratio: float, win_rate: float) -> Dict:
        """
        Calculate overall trade quality score (0-100)
        
        Factors:
        1. Risk/Reward ratio (25%)
        2. Win rate estimate (25%)
        3. Technical setup (20%)
        4. Market regime (15%)
        5. Pattern confirmation (15%)
        """
        scores = {}
        
        # 1. Risk/Reward (max 25 points)
        rr_score = min(risk_reward_ratio / 3.0, 1.0) * 25
        scores['risk_reward'] = rr_score
        
        # 2. Win Rate (max 25 points)
        wr_score = ((win_rate - 30) / 55) * 25  # Scale 30-85% to 0-25
        wr_score = max(0, min(wr_score, 25))
        scores['win_rate'] = wr_score
        
        # 3. Technical Setup (max 20 points)
        tech_score = 0
        
        # RSI
        if 'rsi' in analysis:
            rsi = analysis['rsi']
            if 30 < rsi < 70:
                tech_score += 5
            elif rsi < 30 or rsi > 70:
                tech_score += 3
        
        # MACD
        if 'macd_histogram' in analysis:
            tech_score += 5
        
        # Bollinger Bands
        if all(k in analysis for k in ['current_price', 'bb_upper', 'bb_lower', 'bb_middle']):
            bb_position = (analysis['current_price'] - analysis['bb_lower']) / \
                         (analysis['bb_upper'] - analysis['bb_lower'])
            if 0.2 < bb_position < 0.8:
                tech_score += 5
            else:
                tech_score += 3
        
        # Trend alignment
        if all(k in analysis for k in ['current_price', 'sma_20', 'sma_50']):
            if (analysis['current_price'] > analysis['sma_20'] > analysis['sma_50']) or \
               (analysis['current_price'] < analysis['sma_20'] < analysis['sma_50']):
                tech_score += 5
        
        scores['technical'] = min(tech_score, 20)
        
        # 4. Market Regime (max 15 points)
        regime_score = 0
        if 'regime' in ai_analysis:
            regime = ai_analysis['regime']
            if regime['regime'] in ['trending_up', 'trending_down']:
                regime_score = regime['confidence'] * 15
            elif regime['regime'] == 'ranging':
                regime_score = regime['confidence'] * 10
            else:  # volatile/crisis
                regime_score = regime['confidence'] * 5
        scores['regime'] = regime_score
        
        # 5. Pattern Confirmation (max 15 points)
        pattern_score = 0
        if 'patterns' in ai_analysis and ai_analysis['patterns']:
            patterns = ai_analysis['patterns']
            # Score based on number and strength of patterns
            pattern_strength = np.mean([p.get('strength', 0) for p in patterns])
            pattern_count = min(len(patterns), 3)
            pattern_score = (pattern_strength * pattern_count / 3) * 15
        scores['patterns'] = pattern_score
        
        # Calculate total score
        total_score = sum(scores.values())
        
        return {
            'total_score': total_score,
            'grade': self._grade_trade_quality(total_score),
            'scores': scores,
            'recommendation': self._recommend_action(total_score)
        }
    
    def _grade_trade_quality(self, score: float) -> str:
        """Grade the trade quality"""
        if score >= 80:
            return 'A+ (Excellent)'
        elif score >= 70:
            return 'A (Very Good)'
        elif score >= 60:
            return 'B (Good)'
        elif score >= 50:
            return 'C (Fair)'
        elif score >= 40:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def _recommend_action(self, score: float) -> str:
        """Recommend trading action based on score"""
        if score >= 70:
            return 'Strong Entry - High Confidence'
        elif score >= 60:
            return 'Good Entry - Take Trade'
        elif score >= 50:
            return 'Acceptable Entry - Proceed with Caution'
        elif score >= 40:
            return 'Weak Entry - Consider Waiting'
        else:
            return 'Poor Setup - Do Not Trade'
    
    # ==================== Portfolio Risk ====================
    
    def calculate_portfolio_risk(self, open_positions: List[Dict]) -> Dict:
        """Calculate total portfolio risk"""
        total_risk = 0
        total_exposure = 0
        
        for pos in open_positions:
            # Calculate risk per position
            entry_price = pos.get('price', 0)
            sl = pos.get('sl', 0)
            volume = pos.get('volume', 0)
            
            if sl > 0:
                risk_pips = abs(entry_price - sl)
                # Simplified risk calculation
                risk_amount = risk_pips * volume * 10  # Approximate
                total_risk += risk_amount
            
            # Calculate exposure
            total_exposure += entry_price * volume
        
        portfolio_risk_percent = (total_risk / self.account_balance) * 100 if self.account_balance > 0 else 0
        
        return {
            'total_risk': total_risk,
            'total_exposure': total_exposure,
            'portfolio_risk_percent': portfolio_risk_percent,
            'open_positions': len(open_positions),
            'risk_status': 'Safe' if portfolio_risk_percent < 10 else 'High' if portfolio_risk_percent < 20 else 'Critical'
        }
    
    # ==================== Complete Risk Assessment ====================
    
    def assess_trade_risk(self, symbol: str, order_type: str, entry_price: float,
                         analysis: Dict, ai_analysis: Dict, atr: float) -> Dict:
        """
        Complete risk assessment for a trade
        
        Returns comprehensive risk analysis and recommendations
        """
        # Calculate dynamic SL/TP
        sl_tp = self.calculate_dynamic_sl_tp(symbol, order_type, entry_price, atr)
        
        # Calculate position size
        position_size = self.calculate_position_size(
            symbol, entry_price, sl_tp['stop_loss']
        )
        
        # Calculate risk/reward
        risk_reward = self.calculate_risk_reward(
            entry_price, sl_tp['stop_loss'], sl_tp['take_profit']
        )
        
        # Estimate win rate
        win_rate_est = self.estimate_win_rate(analysis, ai_analysis)
        
        # Calculate expected value
        ev = self.calculate_expected_value(
            win_rate_est['estimated_win_rate'],
            risk_reward['risk_reward_ratio'],
            position_size.get('risk_amount', 0)
        )
        
        # Calculate trade quality score
        quality_score = self.calculate_trade_quality_score(
            analysis, ai_analysis,
            risk_reward['risk_reward_ratio'],
            win_rate_est['estimated_win_rate']
        )
        
        return {
            'symbol': symbol,
            'order_type': order_type,
            'entry_price': entry_price,
            'stop_loss': sl_tp['stop_loss'],
            'take_profit': sl_tp['take_profit'],
            'position_size': position_size,
            'risk_reward': risk_reward,
            'win_rate': win_rate_est,
            'expected_value': ev,
            'quality_score': quality_score,
            'should_take_trade': quality_score['total_score'] >= 50 and ev['profitable'],
            'recommendation': quality_score['recommendation']
        }
    
    # ==================== Advanced Risk Protection 2026 ====================
    
    def check_circuit_breaker(self) -> Dict:
        """
        🛡️ Circuit Breaker: หยุดเทรดเมื่อขาดทุนเกินขีดจำกัด
        
        Returns:
            Dictionary with circuit breaker status
        """
        from datetime import datetime, date, timedelta
        
        # รีเซ็ต daily tracking ถ้าเป็นวันใหม่
        today = date.today()
        if self.last_reset_date != today:
            self.reset_daily_tracking()
        
        # คำนวณ daily loss
        self.daily_loss = self.daily_start_balance - self.account_balance
        daily_loss_percent = (self.daily_loss / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
        
        # คำนวณ drawdown
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance
        self.current_drawdown_percent = ((self.peak_balance - self.account_balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
        
        # เช็คเงื่อนไข Circuit Breaker
        breaker_triggered = False
        reason = []
        
        # 1. Daily Loss Limit
        if daily_loss_percent >= self.daily_loss_limit_percent:
            breaker_triggered = True
            reason.append(f"Daily loss {daily_loss_percent:.1f}% >= {self.daily_loss_limit_percent}%")
        
        # 2. Maximum Drawdown
        if self.current_drawdown_percent >= self.max_drawdown_percent:
            breaker_triggered = True
            reason.append(f"Drawdown {self.current_drawdown_percent:.1f}% >= {self.max_drawdown_percent}%")
        
        # 3. Daily Trade Limit (ถ้าเท่ากับ 0 = ไม่จำกัด)
        if self.max_daily_trades > 0 and self.daily_trade_count >= self.max_daily_trades:
            breaker_triggered = True
            reason.append(f"Daily trades {self.daily_trade_count} >= {self.max_daily_trades}")
        
        self.circuit_breaker_active = breaker_triggered
        
        return {
            'active': breaker_triggered,
            'reason': '; '.join(reason) if reason else 'All limits OK',
            'daily_loss_percent': daily_loss_percent,
            'drawdown_percent': self.current_drawdown_percent,
            'daily_trades': self.daily_trade_count,
            'can_trade': not breaker_triggered
        }
    
    def check_consecutive_loss_protection(self) -> Dict:
        """
        🛡️ Consecutive Loss Protection: ลดความเสี่ยงหลังเสียติดกัน
        
        Returns:
            Adjusted risk percent based on consecutive losses
        """
        if not self.reduce_risk_after_losses:
            return {
                'adjusted_risk': self.risk_percent,
                'consecutive_losses': self.consecutive_losses,
                'reduction_applied': False
            }
        
        # ลดความเสี่ยงตามจำนวนครั้งที่เสียติดกัน
        reduction_factor = 1.0
        
        if self.consecutive_losses >= 5:
            reduction_factor = 0.25  # เหลือ 25%
        elif self.consecutive_losses >= 4:
            reduction_factor = 0.5   # เหลือ 50%
        elif self.consecutive_losses >= 3:
            reduction_factor = 0.75  # เหลือ 75%
        
        adjusted_risk = self.risk_percent * reduction_factor
        adjusted_risk = max(adjusted_risk, self.min_risk_percent)
        
        return {
            'adjusted_risk': adjusted_risk,
            'consecutive_losses': self.consecutive_losses,
            'reduction_factor': reduction_factor,
            'reduction_applied': reduction_factor < 1.0,
            'warning': f"Risk reduced to {reduction_factor*100:.0f}% due to {self.consecutive_losses} consecutive losses" if reduction_factor < 1.0 else None
        }
    
    def check_volatility_filter(self, current_atr: float, symbol: str) -> Dict:
        """
        🛡️ Volatility Filter: ป้องกันการเทรดในช่วงผันผวนสูงผิดปกติ
        
        Args:
            current_atr: Current ATR value
            symbol: Trading symbol
            
        Returns:
            Volatility check results
        """
        # เก็บ history ATR
        self.avg_atr_history.append(current_atr)
        if len(self.avg_atr_history) > 50:  # เก็บ 50 ค่าล่าสุด
            self.avg_atr_history.pop(0)
        
        if len(self.avg_atr_history) < 10:
            return {
                'safe_to_trade': True,
                'volatility_status': 'Normal (Insufficient data)',
                'current_atr': current_atr,
                'avg_atr': current_atr
            }
        
        # คำนวณค่าเฉลี่ย ATR
        avg_atr = np.mean(self.avg_atr_history)
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # เช็คว่า volatility สูงเกินไปหรือไม่
        safe_to_trade = volatility_ratio <= self.max_volatility_multiplier
        
        if volatility_ratio > self.max_volatility_multiplier:
            status = f"⚠️ Extremely High ({volatility_ratio:.1f}x normal)"
        elif volatility_ratio > 2.0:
            status = f"High ({volatility_ratio:.1f}x normal)"
        elif volatility_ratio > 1.5:
            status = f"Elevated ({volatility_ratio:.1f}x normal)"
        else:
            status = f"Normal ({volatility_ratio:.1f}x)"
        
        return {
            'safe_to_trade': safe_to_trade,
            'volatility_status': status,
            'current_atr': current_atr,
            'avg_atr': avg_atr,
            'volatility_ratio': volatility_ratio,
            'warning': f"Market volatility too high! {volatility_ratio:.1f}x normal" if not safe_to_trade else None
        }
    
    def record_trade_result(self, profit_loss: float, was_win: bool):
        """
        บันทึกผลการเทรดเพื่อติดตาม consecutive losses
        
        Args:
            profit_loss: กำไร/ขาดทุนเป็นเงิน
            was_win: True ถ้าชนะ, False ถ้าแพ้
        """
        if was_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # เพิ่มจำนวนเทรดวันนี้
        self.daily_trade_count += 1
        
        # เก็บประวัติ
        self.trade_history.append({
            'timestamp': pd.Timestamp.now(),
            'profit_loss': profit_loss,
            'was_win': was_win,
            'balance_after': self.account_balance
        })
    
    def reset_daily_tracking(self):
        """รีเซ็ตค่าที่ติดตามรายวัน"""
        from datetime import date
        self.last_reset_date = date.today()
        self.daily_start_balance = self.account_balance
        self.daily_loss = 0.0
        self.daily_trade_count = 0
        self.circuit_breaker_active = False
    
    def get_adjusted_risk_percent(self, current_atr: float = None, symbol: str = None) -> Dict:
        """
        🎯 คำนวณความเสี่ยงที่ปรับแล้ว จากทุกปัจจัย
        
        Returns:
            Final adjusted risk percent with all protections applied
        """
        # 1. เริ่มจาก base risk
        final_risk = self.risk_percent
        adjustments = []
        
        # 2. เช็ค Circuit Breaker
        circuit = self.check_circuit_breaker()
        if circuit['active']:
            return {
                'final_risk_percent': 0.0,
                'can_trade': False,
                'reason': f"🔴 CIRCUIT BREAKER: {circuit['reason']}",
                'adjustments': [circuit['reason']]
            }
        
        # 3. เช็ค Consecutive Losses
        loss_protection = self.check_consecutive_loss_protection()
        if loss_protection['reduction_applied']:
            final_risk = loss_protection['adjusted_risk']
            adjustments.append(f"Risk reduced to {loss_protection['reduction_factor']*100:.0f}% ({loss_protection['consecutive_losses']} losses)")
        
        # 4. เช็ค Volatility (ถ้ามีข้อมูล)
        if current_atr is not None and symbol is not None:
            vol_check = self.check_volatility_filter(current_atr, symbol)
            if not vol_check['safe_to_trade']:
                return {
                    'final_risk_percent': 0.0,
                    'can_trade': False,
                    'reason': f"🔴 {vol_check['warning']}",
                    'adjustments': [vol_check['warning']]
                }
            elif vol_check['volatility_ratio'] > 1.5:
                adjustments.append(f"High volatility: {vol_check['volatility_status']}")
        
        # 5. ปรับตาม Drawdown (ยิ่ง drawdown มาก ยิ่งลดความเสี่ยง)
        if self.current_drawdown_percent > 10:
            dd_factor = 1.0 - (self.current_drawdown_percent / 100)
            final_risk *= dd_factor
            adjustments.append(f"Drawdown adjustment: {dd_factor*100:.0f}% ({self.current_drawdown_percent:.1f}% DD)")
        
        # 6. จำกัดไม่ให้ต่ำหรือสูงเกินไป
        final_risk = max(self.min_risk_percent, min(final_risk, self.max_risk_percent))
        
        return {
            'final_risk_percent': final_risk,
            'base_risk_percent': self.risk_percent,
            'can_trade': True,
            'adjustments': adjustments if adjustments else ['No adjustments - normal trading'],
            'protection_status': {
                'circuit_breaker': circuit['active'],
                'consecutive_losses': self.consecutive_losses,
                'drawdown_percent': self.current_drawdown_percent,
                'daily_trades': self.daily_trade_count
            }
        }
    
    def get_risk_protection_summary(self) -> str:
        """
        📊 สรุปสถานะระบบป้องกันความเสี่ยงทั้งหมด
        
        Returns:
            Summary text
        """
        circuit = self.check_circuit_breaker()
        
        summary = "=" * 60 + "\n"
        summary += "🛡️  ADVANCED RISK PROTECTION STATUS (2026)\n"
        summary += "=" * 60 + "\n\n"
        
        summary += f"💰 Account Balance: ${self.account_balance:,.2f}\n"
        summary += f"📈 Peak Balance: ${self.peak_balance:,.2f}\n"
        summary += f"📉 Current Drawdown: {self.current_drawdown_percent:.2f}% (Max: {self.max_drawdown_percent}%)\n"
        summary += f"📅 Daily Loss: ${self.daily_loss:.2f} ({circuit['daily_loss_percent']:.1f}% - Limit: {self.daily_loss_limit_percent}%)\n"
        summary += f"🔢 Daily Trades: {self.daily_trade_count}/{self.max_daily_trades if self.max_daily_trades > 0 else 'ไม่จำกัด'}\n"
        summary += f"❌ Consecutive Losses: {self.consecutive_losses}\n\n"
        
        summary += f"🚦 Circuit Breaker: {'🔴 ACTIVE' if circuit['active'] else '🟢 OK'}\n"
        if circuit['active']:
            summary += f"   Reason: {circuit['reason']}\n"
        
        summary += f"🎯 Trading Status: {'⛔ STOPPED' if circuit['active'] else '✅ ACTIVE'}\n"
        summary += "=" * 60 + "\n"
        
        return summary
