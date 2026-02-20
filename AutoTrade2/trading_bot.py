import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import time
from technical_analysis import TechnicalAnalyzer
from ai_analysis import AIAnalyzer
from risk_manager import RiskManager

class TradingBot:
    def __init__(self, symbols: List[str] = ['EURUSD'], 
                 timeframe: int = mt5.TIMEFRAME_M15,
                 risk_percent: float = 2.0,
                 fixed_volume: float = None):
        self.symbols = symbols
        self.timeframe = timeframe
        self.risk_percent = risk_percent
        self.fixed_volume = fixed_volume  # ถ้ากำหนด = ใช้ fixed volume, ถ้า None = คำนวณจาก risk%
        
        self.technical = TechnicalAnalyzer()
        self.ai = AIAnalyzer()
        self.risk_manager = None  # Will be initialized after getting balance
        
        # Bot settings
        self.min_quality_score = 50  # Minimum trade quality score
        self.min_win_rate = 45  # Minimum win rate estimate
        self.min_rr_ratio = 1.5  # Minimum risk/reward ratio
        self.max_positions = 5  # Maximum open positions
        self.is_running = False
        self.stopping_mode = False  # โหมดกำลังหยุด - รอปิดออเดอร์ก่อน
        
        # 🚀 Fast mode only - เปิดออเดอร์ทันที ปิดเมื่อได้กำไร
        self.aggressive_mode = True  # ใช้โหมดเร็วเท่านั้น
        self.min_profit_pips = 10  # ปิดเมื่อได้กำไรขั้นต่ำ (pips)
        
        # Multi-timeframe settings
        self.use_multi_timeframe = True
        self.confirmation_timeframes = [
            mt5.TIMEFRAME_M15,
            mt5.TIMEFRAME_H1,
            mt5.TIMEFRAME_H4
        ]
        
        # Trade log with size limits
        self.trade_log = []
        self.signals_log = []
        self.max_log_size = 1000  # จำกัดจำนวน logs สูงสุด
        self.max_signals_size = 500  # จำกัดจำนวน signals สูงสุด
        
    def initialize(self, account_balance: float):
        """Initialize risk manager with account balance"""
        self.risk_manager = RiskManager(account_balance, self.risk_percent)
    
    def _cleanup_logs(self):
        """Cleanup old logs to prevent memory issues"""
        try:
            # จำกัด trade_log
            if len(self.trade_log) > self.max_log_size:
                removed = len(self.trade_log) - self.max_log_size
                self.trade_log = self.trade_log[-self.max_log_size:]
                print(f"🧹 Cleaned {removed} old trade records (keeping last {self.max_log_size})")
            
            # จำกัด signals_log
            if len(self.signals_log) > self.max_signals_size:
                removed = len(self.signals_log) - self.max_signals_size
                self.signals_log = self.signals_log[-self.max_signals_size:]
                print(f"🧹 Cleaned {removed} old signal records (keeping last {self.max_signals_size})")
        except Exception as e:
            print(f"⚠️ Error during log cleanup: {e}")
    
    # ==================== Signal Generation ====================
    
    def generate_signal(self, symbol: str) -> Dict:
        """
        Generate trading signal for a symbol
        
        Returns:
            Dictionary with signal and analysis
        """
        try:
            # Get technical analysis
            tech_analysis = self.technical.analyze_complete(symbol, self.timeframe)
            if 'error' in tech_analysis:
                return {'signal': 'NONE', 'error': tech_analysis['error']}
            
            df = tech_analysis['dataframe']
            
            # Get AI analysis
            ai_analysis = self.ai.analyze_complete(df, tech_analysis)
            
            # Multi-timeframe confirmation
            mtf_confirmation = None
            if self.use_multi_timeframe:
                mtf_confirmation = self._check_multi_timeframe_confirmation(symbol)
            
            # Determine signal
            signal = self._determine_signal(tech_analysis, ai_analysis, mtf_confirmation, self.aggressive_mode)
            
            # Get risk assessment if signal is not NONE
            risk_assessment = None
            if signal['direction'] != 'NONE' and self.risk_manager:
                risk_assessment = self.risk_manager.assess_trade_risk(
                    symbol,
                    signal['direction'].lower(),
                    tech_analysis['current_price'],
                    tech_analysis,
                    ai_analysis,
                    tech_analysis['atr']
                )
            
            # Create signal report
            signal_report = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': signal,
                'technical_analysis': tech_analysis,
                'ai_analysis': ai_analysis,
                'mtf_confirmation': mtf_confirmation,
                'risk_assessment': risk_assessment
            }
            
            # Log signal
            self.signals_log.append(signal_report)
            
            # ทำความสะอาด logs ถ้าเยอะเกินไป
            if len(self.signals_log) > self.max_signals_size:
                self._cleanup_logs()
            
            return signal_report
            
        except Exception as e:
            return {
                'signal': 'NONE',
                'error': str(e)
            }
    
    def _determine_signal(self, tech: Dict, ai: Dict, mtf: Optional[Dict], aggressive: bool = False) -> Dict:
        """
        Determine trading signal based on all analysis
        
        Args:
            aggressive: If True, use lower threshold (35 instead of 50)
        
        Returns:
            Dictionary with direction, strength, and reasons
        """
        buy_score = 0
        sell_score = 0
        reasons = {'buy': [], 'sell': []}
        
        # 1. AI Probability (30 points)
        if 'probabilities' in ai:
            prob = ai['probabilities']
            if prob['buy_probability'] > 0.6:
                buy_score += prob['buy_probability'] * 30
                reasons['buy'].append(f"AI: High buy probability {prob['buy_probability']:.1%}")
            elif prob['sell_probability'] > 0.6:
                sell_score += prob['sell_probability'] * 30
                reasons['sell'].append(f"AI: High sell probability {prob['sell_probability']:.1%}")
        
        # 2. Market Regime (20 points)
        if 'regime' in ai:
            regime = ai['regime']
            if regime['regime'] == 'trending_up':
                buy_score += regime['confidence'] * 20
                reasons['buy'].append(f"Trending up market: {regime['confidence']:.2%}")
            elif regime['regime'] == 'trending_down':
                sell_score += regime['confidence'] * 20
                reasons['sell'].append(f"Trending down market: {regime['confidence']:.2%}")
        
        # 3. RSI (15 points - เพิ่มจาก 10)
        if 'rsi' in tech:
            rsi = tech['rsi']
            if rsi < 30:
                buy_score += 15
                reasons['buy'].append(f"RSI oversold: {rsi:.1f}")
            elif rsi < 40:
                buy_score += 8
                reasons['buy'].append(f"RSI low: {rsi:.1f}")
            elif rsi > 70:
                sell_score += 15
                reasons['sell'].append(f"RSI overbought: {rsi:.1f}")
            elif rsi > 60:
                sell_score += 8
                reasons['sell'].append(f"RSI high: {rsi:.1f}")
        
        # 4. MACD (10 points)
        if 'macd_histogram' in tech:
            if tech['macd_histogram'] > 0 and tech['macd'] > tech['macd_signal']:
                buy_score += 10
                reasons['buy'].append("MACD bullish crossover")
            elif tech['macd_histogram'] < 0 and tech['macd'] < tech['macd_signal']:
                sell_score += 10
                reasons['sell'].append("MACD bearish crossover")
        
        # 5. Moving Average Trend (10 points)
        if all(k in tech for k in ['current_price', 'sma_20', 'sma_50', 'ema_12']):
            if tech['current_price'] > tech['ema_12'] > tech['sma_20'] > tech['sma_50']:
                buy_score += 10
                reasons['buy'].append("Strong uptrend - All MAs aligned")
            elif tech['current_price'] < tech['ema_12'] < tech['sma_20'] < tech['sma_50']:
                sell_score += 10
                reasons['sell'].append("Strong downtrend - All MAs aligned")
            elif tech['current_price'] > tech['sma_20'] and tech['sma_20'] > tech['sma_50']:
                buy_score += 5
                reasons['buy'].append("Price above SMA20 (uptrend)")
            elif tech['current_price'] < tech['sma_20'] and tech['sma_20'] < tech['sma_50']:
                sell_score += 5
                reasons['sell'].append("Price below SMA20 (downtrend)")
            # ตรวจจับ reversal: ราคาตัด MA แต่ MA ยัง uptrend/downtrend
            elif tech['current_price'] < tech['sma_20'] and tech['sma_20'] > tech['sma_50']:
                sell_score += 3
                reasons['sell'].append("Price below SMA20 but uptrend weakening")
            elif tech['current_price'] > tech['sma_20'] and tech['sma_20'] < tech['sma_50']:
                buy_score += 3
                reasons['buy'].append("Price above SMA20 but downtrend weakening")
        
        # 6. ADX Trend Strength (5 points)
        if 'adx' in tech and tech['adx'] > 25:
            if tech.get('plus_di', 0) > tech.get('minus_di', 0):
                buy_score += 5
                reasons['buy'].append(f"Strong uptrend ADX: {tech['adx']:.1f}")
            else:
                sell_score += 5
                reasons['sell'].append(f"Strong downtrend ADX: {tech['adx']:.1f}")
        
        # 7. Bollinger Bands (8 points - เพิ่มจาก 5)
        if all(k in tech for k in ['current_price', 'bb_upper', 'bb_lower', 'bb_middle']):
            bb_range = tech['bb_upper'] - tech['bb_lower']
            if bb_range > 0:  # ป้องกัน division by zero
                bb_position = (tech['current_price'] - tech['bb_lower']) / bb_range
                if bb_position < 0.2:
                    buy_score += 8
                    reasons['buy'].append(f"Price near lower BB ({bb_position:.1%})")
                elif bb_position < 0.35:
                    buy_score += 4
                    reasons['buy'].append(f"Price in lower BB zone ({bb_position:.1%})")
                elif bb_position > 0.8:
                    sell_score += 8
                    reasons['sell'].append(f"Price near upper BB ({bb_position:.1%})")
                elif bb_position > 0.65:
                    sell_score += 4
                    reasons['sell'].append(f"Price in upper BB zone ({bb_position:.1%})")
        
        # 8. Patterns (12 points - เพิ่มจาก 10)
        if 'patterns' in ai and ai['patterns']:
            for pattern in ai['patterns'][-3:]:  # Last 3 patterns
                if pattern.get('signal') == 'bullish':
                    pattern_score = pattern.get('strength', 0) * 4  # เพิ่มจาก 3
                    buy_score += pattern_score
                    reasons['buy'].append(f"{pattern.get('pattern', 'Pattern')}: {pattern.get('strength', 0):.1%}")
                elif pattern.get('signal') == 'bearish':
                    pattern_score = pattern.get('strength', 0) * 4
                    sell_score += pattern_score
                    reasons['sell'].append(f"{pattern.get('pattern', 'Pattern')}: {pattern.get('strength', 0):.1%}")
        
        # 9. Multi-Timeframe Confirmation (10 points - bonus)
        if mtf and mtf['confirmation']:
            if mtf['direction'] == 'BUY':
                buy_score += 10
                reasons['buy'].append(f"Multi-timeframe confirmation: {mtf['strength']:.1f}%")
            elif mtf['direction'] == 'SELL':
                sell_score += 10
                reasons['sell'].append(f"Multi-timeframe confirmation: {mtf['strength']:.1f}%")
        
        # 10. ตรวจจับ Reversal Conditions (10 points)
        # ตรวจหาสัญญาณ overbought/oversold + bearish/bullish pattern
        if 'rsi' in tech and 'patterns' in ai and ai['patterns']:
            rsi = tech['rsi']
            recent_patterns = [p.get('signal') for p in ai['patterns'][-2:]]
            
            # Overbought + bearish pattern = strong sell
            if rsi > 65 and 'bearish' in recent_patterns:
                sell_score += 10
                reasons['sell'].append(f"Reversal: Overbought RSI {rsi:.1f} + bearish pattern")
            # Oversold + bullish pattern = strong buy
            elif rsi < 35 and 'bullish' in recent_patterns:
                buy_score += 10
                reasons['buy'].append(f"Reversal: Oversold RSI {rsi:.1f} + bullish pattern")
        
        # Determine direction with threshold
        threshold = 35 if aggressive else 50
        
        if buy_score > sell_score and buy_score >= threshold:
            direction = 'BUY'
            strength = min(buy_score, 100)
            signal_reasons = reasons['buy']
        elif sell_score > buy_score and sell_score >= threshold:
            direction = 'SELL'
            strength = min(sell_score, 100)
            signal_reasons = reasons['sell']
        else:
            direction = 'NONE'
            strength = 0
            signal_reasons = ['Signal not strong enough']
        
        return {
            'direction': direction,
            'strength': strength,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'reasons': signal_reasons,
            'buy_reasons': reasons['buy'],  # เก็บ buy reasons ไว้แสดง
            'sell_reasons': reasons['sell'],  # เก็บ sell reasons ไว้แสดง
            'confidence': abs(buy_score - sell_score)
        }
    
    def _check_multi_timeframe_confirmation(self, symbol: str) -> Dict:
        """Check for multi-timeframe trend confirmation"""
        confirmations = []
        
        for tf in self.confirmation_timeframes:
            try:
                df = self.technical.get_ohlcv_data(symbol, tf, 100)
                if df is None:
                    continue
                
                close = df['close']
                sma_20 = close.rolling(window=20).mean()
                sma_50 = close.rolling(window=50).mean()
                
                # Check trend
                if close.iloc[-1] > sma_20.iloc[-1] > sma_50.iloc[-1]:
                    confirmations.append({'timeframe': tf, 'trend': 'BUY'})
                elif close.iloc[-1] < sma_20.iloc[-1] < sma_50.iloc[-1]:
                    confirmations.append({'timeframe': tf, 'trend': 'SELL'})
                else:
                    confirmations.append({'timeframe': tf, 'trend': 'NONE'})
            except:
                continue
        
        # Check if all timeframes agree
        if len(confirmations) >= 2:
            buy_count = sum(1 for c in confirmations if c['trend'] == 'BUY')
            sell_count = sum(1 for c in confirmations if c['trend'] == 'SELL')
            
            if buy_count >= 2:
                return {
                    'confirmation': True,
                    'direction': 'BUY',
                    'strength': (buy_count / len(confirmations)) * 100,
                    'details': confirmations
                }
            elif sell_count >= 2:
                return {
                    'confirmation': True,
                    'direction': 'SELL',
                    'strength': (sell_count / len(confirmations)) * 100,
                    'details': confirmations
                }
        
        return {
            'confirmation': False,
            'direction': 'NONE',
            'strength': 0,
            'details': confirmations
        }
    
    # ==================== Trade Execution ====================
    
    def should_take_trade(self, signal_report: Dict) -> Tuple[bool, str]:
        """
        Determine if trade should be executed based on criteria
        
        Returns:
            (should_trade, reason)
        """
        if signal_report['signal']['direction'] == 'NONE':
            return False, "No clear signal"
        
        # 🚀 FAST MODE - เปิดออเดอร์ทันที ไม่เช็คคุณภาพ
        # เช็คแค่จำนวน positions
        positions = mt5.positions_get()
        current_positions = len(positions) if positions else 0
        if positions is not None and current_positions >= self.max_positions:
            return False, f"🚀 เปิดออเดอร์ครบแล้ว ({current_positions}/{self.max_positions})"
        
        signal = signal_report['signal']
        return True, f"🚀 เปิด {signal['direction']} ทันที!"
    
    def execute_trade(self, signal_report: Dict, trader) -> Dict:
        """
        Execute trade based on signal
        
        Args:
            signal_report: Signal report from generate_signal()
            trader: MT5Trader instance
            
        Returns:
            Trade execution result
        """
        try:
            # Check if should take trade
            should_trade, reason = self.should_take_trade(signal_report)
            
            if not should_trade:
                return {
                    'success': False,
                    'reason': reason,
                    'action': 'skipped'
                }
            
            # Get trade details
            symbol = signal_report['symbol']
            direction = signal_report['signal']['direction'].lower()
            risk = signal_report['risk_assessment']
            
            # 📊 ตรวจสอบและคำนวณ position size
            if self.fixed_volume is not None:
                # ใช้ Fixed Volume ที่ผู้ใช้กำหนด
                calculated_lots = self.fixed_volume
                print(f"\n📊 ใช้ Fixed Volume: {calculated_lots:.2f} lots")
            else:
                # คำนวณจาก risk% (เหมือนเดิม)
                calculated_lots = risk['position_size']['lot_size']
                print(f"\n📊 คำนวณจาก Risk {self.risk_percent}%: {calculated_lots:.2f} lots")
            
            # จำกัด position size สูงสุด (ป้องกัน margin blow-up)
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                max_safe_lots = min(1.0, symbol_info.volume_max)  # จำกัดไม่เกิน 1.0 lot
                
                print(f"   ℹ️  Symbol Info - Min: {symbol_info.volume_min}, Max: {symbol_info.volume_max}")
                
                if calculated_lots > max_safe_lots:
                    print(f"\n⚠️  Position size ใหญ่เกิน: {calculated_lots:.2f} lots")
                    print(f"   จำกัดเหลือ: {max_safe_lots:.2f} lots (ปลอดภัย)")
                    actual_lots = max_safe_lots
                elif calculated_lots < symbol_info.volume_min:
                    print(f"\n⚠️  Volume ต่ำกว่าขั้นต่ำของโบรกเกอร์: {calculated_lots:.2f} < {symbol_info.volume_min}")
                    print(f"   ใช้ค่าต่ำสุดแทน: {symbol_info.volume_min} lots")
                    actual_lots = symbol_info.volume_min
                else:
                    actual_lots = calculated_lots
                    print(f"   ✅ ใช้ Volume: {actual_lots:.2f} lots")
            else:
                # ถ้าไม่มี symbol info ใช้ 0.01 (ปลอดภัยที่สุด)
                print(f"\n⚠️  ไม่สามารถดึงข้อมูล Symbol ได้ - ใช้ค่าเริ่มต้น 0.01 lots")
                actual_lots = 0.01
            
            # Place order โดยไม่กำหนด S/L และ T/P (ให้ bot สแกนแล้วปิดเอง)
            result = trader.place_order(
                symbol=symbol,
                order_type=direction,
                volume=actual_lots,  # ใช้ volume ที่ผ่าน validation แล้ว
                sl_pips=None,  # ไม่กำหนด S/L
                tp_pips=None,  # ไม่กำหนด T/P
                comment=f"AutoBot-Q{risk['quality_score']['total_score']:.0f}-NoSLTP"
            )
            
            # Log trade
            if result.get('success'):
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': result.get('price'),
                    'lot_size': result.get('volume'),
                    'stop_loss': risk['stop_loss'],
                    'take_profit': risk['take_profit'],
                    'quality_score': risk['quality_score']['total_score'],
                    'win_rate': risk['win_rate']['estimated_win_rate'],
                    'rr_ratio': risk['risk_reward']['risk_reward_ratio'],
                    'expected_value': risk['expected_value']['ev_percent'],
                    'order_id': result.get('order'),
                    'signal_strength': signal_report['signal']['strength'],
                    'reasons': signal_report['signal']['reasons']
                }
                self.trade_log.append(trade_record)
                
                # ทำความสะอาด logs ถ้าเยอะเกินไป
                if len(self.trade_log) > self.max_log_size:
                    self._cleanup_logs()
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def monitor_positions_for_profit(self, trader) -> Dict:
        """
        ตรวจสอบออเดอร์ที่เปิดอยู่ และปิดเมื่อมีกำไร (ไม่ว่าจะกี่ pips)
        
        Returns:
            Dictionary with monitoring results
        """
        try:
            positions = mt5.positions_get()
            if not positions or len(positions) == 0:
                return {'monitored': 0, 'closed': 0}
            
            closed_count = 0
            
            for pos in positions:
                symbol = pos.symbol
                ticket = pos.ticket
                order_type = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
                
                # Get current price
                symbol_info = mt5.symbol_info_tick(symbol)
                if not symbol_info:
                    continue
                
                current_price = symbol_info.bid if order_type == 'buy' else symbol_info.ask
                entry_price = pos.price_open
                
                # คำนวณกำไร/ขาดทุนเป็น pips (สำหรับแสดงผล)
                point = mt5.symbol_info(symbol).point
                pip_size = point * 10 if mt5.symbol_info(symbol).digits in [3, 5] else point
                
                if order_type == 'buy':
                    profit_pips = (current_price - entry_price) / pip_size
                else:
                    profit_pips = (entry_price - current_price) / pip_size
                
                profit_money = pos.profit
                
                should_close = False
                close_reason = ""
                
                # ⚡ ปิดออเดอร์ทันทีเมื่อมีกำไร (ไม่ว่ากี่ pips)
                if profit_money > 0:
                    should_close = True
                    close_reason = "profit"
                    print(f"\n💰 มีกำไรแล้ว! กำลังปิดออเดอร์...")
                    print(f"   Ticket: {ticket}")
                    print(f"   Symbol: {symbol}")
                    print(f"   Type: {order_type.upper()}")
                    print(f"   กำไร: {profit_pips:.2f} pips (${profit_money:.2f})")
                
                # 🛡️ ปิดออเดอร์เมื่อขาดทุนเกินกำหนด (Stop Loss)
                elif self.risk_manager.max_loss_per_trade > 0 and profit_money <= -self.risk_manager.max_loss_per_trade:
                    should_close = True
                    close_reason = "stop_loss"
                    print(f"\n🛑 ขาดทุนเกินกำหนด! กำลังปิดออเดอร์...")
                    print(f"   Ticket: {ticket}")
                    print(f"   Symbol: {symbol}")
                    print(f"   Type: {order_type.upper()}")
                    print(f"   ขาดทุน: {profit_pips:.2f} pips (${profit_money:.2f})")
                    print(f"   จำกัดขาดทุน: ${self.risk_manager.max_loss_per_trade:.2f}")
                
                # ปิดออเดอร์ถ้าตรงเงื่อนไข
                if should_close:
                    # ปิดออเดอร์
                    result = trader.close_position(ticket)
                    
                    if result.get('success'):
                        closed_count += 1
                        print(f"   ✅ ปิดออเดอร์สำเร็จ!")
                        
                        # 🛡️ บันทึกผลการเทรด
                        if self.risk_manager:
                            was_win = (close_reason == "profit")
                            self.risk_manager.record_trade_result(profit_money, was_win=was_win)
                    else:
                        print(f"   ❌ ปิดออเดอร์ล้มเหลว: {result.get('error')}")
            
            return {
                'monitored': len(positions),
                'closed': closed_count
            }
            
        except Exception as e:
            print(f"❌ Error monitoring positions: {e}")
            return {'monitored': 0, 'closed': 0, 'error': str(e)}
    
    # ==================== Auto Trading ====================
    
    def start_auto_trading(self, trader, interval_seconds: int = 60):
        """
        Start auto trading bot
        
        Args:
            trader: MT5Trader instance
            interval_seconds: Seconds between scans
        """
        # 🔄 Reset flags เมื่อเริ่มใหม่
        if self.stopping_mode:
            print("ℹ️  ยกเลิกโหมดหยุดก่อนหน้า - เตรียมเริ่มบอทใหม่")
        
        self.is_running = True
        self.stopping_mode = False  # Reset stopping mode
        
        print("🤖 Auto Trading Bot Started!")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"⏱️ Scanning every {interval_seconds} seconds")
        print(f"🚀 MODE: FAST (เปิดออเดอร์ทันที - ปิดเมื่อได้กำไร)")
        print(f"💰 ปิดออเดอร์ทันทีเมื่อมีกำไร (ไม่ว่ากี่ pips)")
        print(f"⚠️  ไม่มี S/L และ T/P - ต้องสแกนตลอดเวลา!")
        print(f"💰 Risk per Trade: {self.risk_percent}%")
        print("-" * 60)
        
        # ⚠️ คำเตือนสำคัญ
        print("\n⚠️  คำเตือนสำคัญ:")
        print("   🚫 ไม่มี Stop Loss (S/L) และ Take Profit (T/P)")
        print("   💰 จะปิดออเดอร์เมื่อมีกำไร > 0 เท่านั้น")
        print("   📊 ต้องรัน bot ตลอดเวลาเพื่อสแกนและปิดออเดอร์")
        print("   ⚠️  ถ้า bot หยุดทำงาน ออเดอร์จะไม่ถูกปิดอัตโนมัติ!")
        print("   🛡️  แนะนำ: ติดตาม Drawdown Protection อย่างใกล้ชิด")
        print("-" * 60)
        
        # 🛡️ แสดงสถานะระบบป้องกัน
        print("\n🛡️  ADVANCED RISK PROTECTION (2026):")
        print(f"   ⛔ Daily Loss Limit: {self.risk_manager.daily_loss_limit_percent}%")
        print(f"   ⛔ Max Drawdown: {self.risk_manager.max_drawdown_percent}%")
        
        daily_trades_text = f"{self.risk_manager.max_daily_trades} ครั้ง" if self.risk_manager.max_daily_trades > 0 else "ไม่จำกัด"
        print(f"   ⛔ Max Daily Trades: {daily_trades_text}")
        
        print(f"   ⛔ Max Consecutive Losses: {self.risk_manager.max_consecutive_losses}")
        
        max_loss_text = f"${self.risk_manager.max_loss_per_trade:.2f}" if self.risk_manager.max_loss_per_trade > 0 else "ไม่จำกัด"
        print(f"   ⛔ Max Loss Per Trade: {max_loss_text}")
        
        print(f"   ⛔ Max Volatility: {self.risk_manager.max_volatility_multiplier}x normal")
        
        print("\n⚠️  CIRCUIT BREAKERS:")
        print("   Bot will auto-stop if:")
        print("   • Daily loss reaches limit")
        print("   • Drawdown exceeds maximum")
        print("   • Too many trades in one day")
        print("   • Market volatility too high")
        print("   ")
        print("   Orders will auto-close when:")
        print("   • Profit > $0 (take profit)")
        if self.risk_manager.max_loss_per_trade > 0:
            print(f"   • Loss >= ${self.risk_manager.max_loss_per_trade:.2f} (stop loss)")
        print("   → Protection active to safeguard your capital!")
        
        print("-" * 60)
        
        # Track connection errors
        connection_errors = 0
        max_connection_errors = 3
        
        # Track scans for periodic cleanup
        scan_counter = 0
        cleanup_interval = 100  # ทำความสะอาดทุก 100 รอบ (ประมาณ 8-10 นาที)
        
        while self.is_running:
            try:
                # 🔌 เช็คการเชื่อมต่อ MT5
                if not trader.check_connection():
                    print(f"\n⚠️  MT5 connection lost! Attempting to reconnect...")
                    
                    # พยายาม reconnect
                    if trader.reconnect():
                        print("✅ Reconnection successful!")
                        connection_errors = 0  # Reset counter
                    else:
                        connection_errors += 1
                        print(f"❌ Reconnection failed ({connection_errors}/{max_connection_errors})")
                        
                        if connection_errors >= max_connection_errors:
                            print(f"\n🛑 TOO MANY CONNECTION FAILURES!")
                            print(f"   Bot will stop after {max_connection_errors} failed attempts")
                            print(f"   Please check your MT5 terminal and restart the bot")
                            self.is_running = False
                            break
                        
                        print(f"⏳ Waiting {interval_seconds} seconds before retry...")
                        time.sleep(interval_seconds)
                        continue
                
                # Update account balance
                account_info = trader.get_account_info()
                if account_info:
                    self.risk_manager.update_balance(account_info['balance'])
                
                # �️ เช็คระบบป้องกันความเสี่ยงขั้นสูง 2026
                protection_check = self.risk_manager.get_adjusted_risk_percent()
                
                if not protection_check['can_trade']:
                    print(f"\n{'='*60}")
                    print(f"🛑 TRADING HALTED!")
                    print(f"Reason: {protection_check['reason']}")
                    print(self.risk_manager.get_risk_protection_summary())
                    print(f"⏸️  Waiting until next check...")
                    print(f"{'='*60}")
                    time.sleep(interval_seconds)
                    continue
                
                # แสดงสถานะการป้องกัน
                if protection_check['adjustments'] and protection_check['adjustments'][0] != 'No adjustments - normal trading':
                    print(f"\n🛡️  Risk Protection Active:")
                    for adj in protection_check['adjustments']:
                        print(f"   - {adj}")
                    print(f"   Adjusted Risk: {protection_check['final_risk_percent']:.2f}%")
                
                # อัพเดทความเสี่ยงที่ปรับแล้ว
                self.risk_manager.risk_percent = protection_check['final_risk_percent']
                
                # 🧹 ทำความสะอาด memory เป็นระยะ
                scan_counter += 1
                if scan_counter >= cleanup_interval:
                    self._cleanup_logs()
                    scan_counter = 0  # Reset counter
                    print(f"ℹ️  Memory cleanup completed (every {cleanup_interval} scans)")
                
                # 💰 ตรวจสอบและปิดออเดอร์ที่มีกำไรแล้ว (ทุกโหมด)
                monitor_result = self.monitor_positions_for_profit(trader)
                if monitor_result.get('closed', 0) > 0:
                    print(f"✅ ปิดออเดอร์ได้ {monitor_result['closed']} ออเดอร์ที่มีกำไร")
                
                # 🛑 ถ้าอยู่ใน stopping_mode - รอให้ออเดอร์หมดก่อนหยุด
                if self.stopping_mode:
                    positions = mt5.positions_get()
                    current_positions = len(positions) if positions else 0
                    
                    if current_positions == 0:
                        print("\n✅ ปิดออเดอร์ทั้งหมดเรียบร้อยแล้ว")
                        print("🛑 หยุดบอทเรียบร้อย")
                        self.is_running = False
                        break
                    else:
                        print(f"\n⏳ กำลังรอปิดออเดอร์... (เหลืออีก {current_positions} ออเดอร์)")
                        print(f"   💡 บอทจะหยุดเมื่อปิดออเดอร์ทั้งหมดที่มีกำไรแล้ว")
                        time.sleep(interval_seconds)
                        continue
                
                # Scan each symbol (ถ้าไม่ได้อยู่ใน stopping_mode)
                for symbol in self.symbols:
                    # แสดงสถานะปัจจุบัน
                    positions = mt5.positions_get()
                    current_positions = len(positions) if positions else 0
                    mode_text = "🚀 Fast"
                    
                    print(f"\n🔍 Scanning {symbol}...")
                    print(f"   Mode: {mode_text} | Positions: {current_positions}/{self.max_positions}")
                    
                    # Generate signal
                    signal_report = self.generate_signal(symbol)
                    
                    if 'error' in signal_report:
                        print(f"❌ Error: {signal_report['error']}")
                        continue
                    
                    signal = signal_report['signal']
                    
                    # แสดงผลพื้นฐาน
                    print(f"📊 Signal: {signal['direction']} ", end='')
                    print(f"(Strength: {signal['strength']:.1f}%, Confidence: {signal['confidence']:.1f}%)")
                    print(f"   💚 Buy Score: {signal['buy_score']:.1f}/100")
                    print(f"   🔴 Sell Score: {signal['sell_score']:.1f}/100")
                    
                    # แสดงรายละเอียด
                    if signal['direction'] == 'NONE':
                        threshold = 35
                        print(f"\n   ℹ️  ไม่มีสัญญาณชัดเจน (🚀 Fast Mode):")
                        print(f"      - Buy score: {signal['buy_score']:.1f} (ต้องการ ≥{threshold})")
                        print(f"      - Sell score: {signal['sell_score']:.1f} (ต้องการ ≥{threshold})")
                        print(f"      - Confidence: {signal['confidence']:.1f}% (ความแตกต่างน้อย)")
                        
                        # แสดง factors ที่มี (ถ้ามี)
                        if signal.get('buy_reasons'):
                            print(f"\n   💚 Buy Factors ({len(signal['buy_reasons'])}):")
                            for reason in signal['buy_reasons'][:3]:
                                print(f"      + {reason}")
                        if signal.get('sell_reasons'):
                            print(f"\n   🔴 Sell Factors ({len(signal['sell_reasons'])}):")
                            for reason in signal['sell_reasons'][:3]:
                                print(f"      - {reason}")
                        
                        if not signal.get('buy_reasons') and not signal.get('sell_reasons'):
                            print(f"\n   ⚠️  ตลาดไม่มีทิศทางชัด (sideways/ranging)")
                        
                        # 🚫 แจ้งเหตุผลที่ไม่สั่งออเดอร์
                        threshold = 35
                        print(f"\n   🚫 ไม่สั่งออเดอร์เพราะ:")
                        if signal['buy_score'] < threshold and signal['sell_score'] < threshold:
                            max_score = max(signal['buy_score'], signal['sell_score'])
                            direction = "BUY" if signal['buy_score'] > signal['sell_score'] else "SELL"
                            needed = threshold - max_score
                            print(f"      ⚡ ขาดอีก {needed:.1f} คะแนนสำหรับสัญญาณ {direction}")
                            print(f"      💡 ต้องการ indicator เพิ่มเติมเพื่อยืนยันแนวโน้ม")
                            print(f"      🚀 Fast Mode threshold: {threshold}")
                            print(f"      ⏳ รอจนกว่าคะแนนจะถึง {threshold} หรือมากกว่า")
                        
                        elif abs(signal['buy_score'] - signal['sell_score']) < 20:
                            print(f"      ⚖️  ตลาดไม่แน่นอน - Buy และ Sell ใกล้เคียงกัน")
                            print(f"      ⏳ รอจนกว่าจะมีทิศทางชัดเจนขึ้น")
                    else:
                        print(f"\n💡 Main Reasons:")
                        for i, reason in enumerate(signal['reasons'][:5], 1):
                            print(f"   {i}. {reason}")
                        
                        # Check risk assessment
                        risk = signal_report.get('risk_assessment')
                        if risk:
                            print(f"\n💼 Risk Assessment:")
                            print(f"   Quality Score: {risk['quality_score']['total_score']:.1f}/100 "
                                  f"({risk['quality_score']['grade']})")
                            print(f"   Win Rate: {risk['win_rate']['estimated_win_rate']:.1f}%")
                            print(f"   R/R Ratio: {risk['risk_reward']['risk_reward_ratio']:.2f}")
                            print(f"   Expected Value: {risk['expected_value']['ev_percent']:.2f}%")
                            print(f"   Position Size: {risk['position_size']['lot_size']} lots")
                            
                            # Execute trade if criteria met
                            should_trade, reason = self.should_take_trade(signal_report)
                            
                            if should_trade:
                                print(f"\n✅ {reason}")
                                print(f"🚀 Executing {signal['direction']} trade...")
                                
                                result = self.execute_trade(signal_report, trader)
                                
                                if result.get('success'):
                                    print(f"✅ Trade executed successfully!")
                                    print(f"   Order ID: {result.get('order')}")
                                else:
                                    print(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                            else:
                                print(f"\n⏭️ Skipping trade: {reason}")
                
                # Wait before next scan
                print(f"\n⏳ Waiting {interval_seconds} seconds until next scan...")
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\n⚠️ Auto trading stopped by user")
                self.is_running = False
                break
            except Exception as e:
                print(f"\n❌ Error in auto trading loop: {e}")
                time.sleep(interval_seconds)
        
        print("\n🛑 Auto Trading Bot Stopped")
    
    def stop_auto_trading(self):
        """Stop auto trading - will wait to close all positions first"""
        if self.is_running:
            print("\n🛑 กำลังเตรียมหยุดบอท...")
            print("⏳ รอปิดออเดอร์ทั้งหมดที่มีกำไรก่อน...")
            self.stopping_mode = True
    
    # ==================== Analysis Summary ====================
    
    def get_signal_summary(self, signal_report: Dict) -> str:
        """Get formatted signal summary"""
        if 'error' in signal_report:
            return f"Error: {signal_report['error']}"
        
        signal = signal_report['signal']
        tech = signal_report.get('technical_analysis', {})
        ai = signal_report.get('ai_analysis', {})
        risk = signal_report.get('risk_assessment')
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    TRADING SIGNAL ANALYSIS                      ║
╠══════════════════════════════════════════════════════════════╣
║ Symbol: {signal_report['symbol']:<20} Time: {signal_report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
║ Signal: {signal['direction']:<10} Strength: {signal['strength']:.1f}%  Confidence: {signal['confidence']:.1f}%
╠══════════════════════════════════════════════════════════════╣
║ TECHNICAL INDICATORS:
║ • RSI: {tech.get('rsi', 0):.1f}  MACD: {tech.get('macd_histogram', 0):.4f}
║ • ADX: {tech.get('adx', 0):.1f}  ATR: {tech.get('atr', 0):.5f}
║ • SMA20: {tech.get('sma_20', 0):.5f}  SMA50: {tech.get('sma_50', 0):.5f}
║ • Price: {tech.get('current_price', 0):.5f}
╠══════════════════════════════════════════════════════════════╣
║ AI ANALYSIS:
║ • Market Regime: {ai.get('regime', {}).get('regime', 'N/A'):<15} ({ai.get('regime', {}).get('confidence', 0):.1%})
║ • Patterns Found: {ai.get('pattern_count', 0)}
║ • Buy Probability: {ai.get('probabilities', {}).get('buy_probability', 0):.1%}
║ • Sell Probability: {ai.get('probabilities', {}).get('sell_probability', 0):.1%}
╠══════════════════════════════════════════════════════════════╣
"""
        
        if risk:
            summary += f"""║ RISK ASSESSMENT:
║ • Quality Score: {risk['quality_score']['total_score']:.1f}/100 - {risk['quality_score']['grade']}
║ • Win Rate Est: {risk['win_rate']['estimated_win_rate']:.1f}%
║ • R/R Ratio: {risk['risk_reward']['risk_reward_ratio']:.2f}
║ • Expected Value: {risk['expected_value']['ev_percent']:.2f}%
║ • Position Size: {risk['position_size']['lot_size']:.2f} lots
║ • Risk Amount: ${risk['position_size'].get('risk_amount', 0):.2f}
║ • Stop Loss: {risk['stop_loss']:.5f} ({risk['risk_reward']['risk']:.5f})
║ • Take Profit: {risk['take_profit']:.5f} ({risk['risk_reward']['reward']:.5f})
║ 
║ Recommendation: {risk['recommendation']}
╠══════════════════════════════════════════════════════════════╣
"""
        
        summary += f"""║ REASONS:
"""
        for i, reason in enumerate(signal['reasons'][:5], 1):
            summary += f"║ {i}. {reason}\n"
        
        summary += "╚══════════════════════════════════════════════════════════════╝"
        
        return summary
    
    def get_trade_statistics(self) -> Dict:
        """Get trading statistics"""
        if not self.trade_log:
            return {'error': 'No trades executed yet'}
        
        total_trades = len(self.trade_log)
        avg_quality = np.mean([t['quality_score'] for t in self.trade_log])
        avg_win_rate = np.mean([t['win_rate'] for t in self.trade_log])
        avg_rr = np.mean([t['rr_ratio'] for t in self.trade_log])
        avg_ev = np.mean([t['expected_value'] for t in self.trade_log])
        
        buy_trades = sum(1 for t in self.trade_log if t['direction'] == 'buy')
        sell_trades = total_trades - buy_trades
        
        return {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'avg_quality_score': avg_quality,
            'avg_win_rate': avg_win_rate,
            'avg_rr_ratio': avg_rr,
            'avg_expected_value': avg_ev,
            'last_trade': self.trade_log[-1] if self.trade_log else None
        }
