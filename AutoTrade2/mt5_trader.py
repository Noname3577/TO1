"""
MT5 Trader Module
Handles all MetaTrader 5 trading operations
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time

class MT5Trader:
    def __init__(self):
        """Initialize MT5 Trader"""
        self.connected = False
        
    def connect(self, login=None, password=None, server=None):
        """
        Connect to MT5 terminal
        
        Args:
            login: Account login (optional, uses current logged-in account if None)
            password: Account password (optional)
            server: Server name (optional)
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize MT5 connection
            if login and password and server:
                initialized = mt5.initialize(
                    login=login,
                    password=password,
                    server=server
                )
            else:
                # Connect to already logged-in MT5 terminal
                initialized = mt5.initialize()
            
            if not initialized:
                print(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                mt5.shutdown()
                return False
            
            self.connected = True
            print(f"Connected to MT5 - Account: {account_info.login}")
            return True
            
        except Exception as e:
            print(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Disconnected from MT5")
    
    def check_connection(self):
        """
        Check if MT5 connection is still active
        
        Returns:
            bool: True if connected and working
        """
        try:
            if not self.connected:
                return False
            
            # Try to get account info to verify connection
            account_info = mt5.account_info()
            if account_info is None:
                self.connected = False
                return False
            
            return True
        except Exception as e:
            self.connected = False
            return False
    
    def reconnect(self):
        """
        Attempt to reconnect to MT5
        
        Returns:
            bool: True if reconnection successful
        """
        try:
            print("🔄 Attempting to reconnect to MT5...")
            
            # Shutdown existing connection if any
            try:
                mt5.shutdown()
            except:
                pass
            
            self.connected = False
            
            # Try to initialize connection
            initialized = mt5.initialize()
            
            if not initialized:
                print(f"❌ Reconnection failed: {mt5.last_error()}")
                return False
            
            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                print("❌ Reconnection failed: Cannot get account info")
                mt5.shutdown()
                return False
            
            self.connected = True
            print(f"✅ Reconnected to MT5 - Account: {account_info.login}")
            return True
            
        except Exception as e:
            print(f"❌ Reconnection error: {str(e)}")
            return False
    
    def get_account_info(self):
        """
        Get account information
        
        Returns:
            dict: Account information including balance, equity, margin, etc.
        """
        if not self.connected:
            return None
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'free_margin': account_info.margin_free,
                'profit': account_info.profit,
                'leverage': account_info.leverage,
                'currency': account_info.currency
            }
        except Exception as e:
            print(f"Error getting account info: {str(e)}")
            return None
    
    def get_symbol_info(self, symbol):
        """
        Get symbol information
        
        Args:
            symbol: Symbol name (e.g., 'EURUSD')
            
        Returns:
            dict: Symbol information
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            # Make sure symbol is visible in Market Watch
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return None
            
            return {
                'name': symbol_info.name,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_contract_size': symbol_info.trade_contract_size,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step
            }
        except Exception as e:
            print(f"Error getting symbol info: {str(e)}")
            return None
    
    def get_symbol_price(self, symbol):
        """
        Get current symbol price
        
        Args:
            symbol: Symbol name
            
        Returns:
            dict: Bid and ask prices
        """
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time)
            }
        except Exception as e:
            print(f"Error getting symbol price: {str(e)}")
            return None
    
    def get_filling_mode(self, symbol):
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return mt5.ORDER_FILLING_FOK
            
            filling_modes = symbol_info.filling_mode
            
            # Try ORDER_FILLING_FOK first (Fill or Kill)
            if filling_modes & 0x01:  # FOK
                return mt5.ORDER_FILLING_FOK
            # Try ORDER_FILLING_IOC (Immediate or Cancel)
            elif filling_modes & 0x02:  # IOC
                return mt5.ORDER_FILLING_IOC
            # Try ORDER_FILLING_RETURN
            elif filling_modes & 0x04:  # RETURN
                return mt5.ORDER_FILLING_RETURN
            else:
                # Default to FOK
                return mt5.ORDER_FILLING_FOK
                
        except Exception as e:
            print(f"Error getting filling mode: {str(e)}")
            return mt5.ORDER_FILLING_FOK
    
    def place_order(self, symbol, order_type, volume, sl_pips=0, tp_pips=0, 
                    comment="Python AutoTrader", magic=234000):
        """
        Place a market order (Buy or Sell)
        
        Args:
            symbol: Symbol name (e.g., 'EURUSD')
            order_type: 'buy' or 'sell'
            volume: Lot size (e.g., 0.01)
            sl_pips: Stop loss in pips (optional)
            tp_pips: Take profit in pips (optional)
            comment: Order comment
            magic: Magic number for order identification
            
        Returns:
            dict: Order result with success status and order ticket
        """
        if not self.connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
            
            # Make sure symbol is visible
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {'success': False, 'error': f'Failed to select symbol {symbol}'}
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': 'Failed to get current price'}
            
            # Determine order type and price
            if order_type.lower() == 'buy':
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl_price = price - (sl_pips * symbol_info.point * 10) if (sl_pips and sl_pips > 0) else 0
                tp_price = price + (tp_pips * symbol_info.point * 10) if (tp_pips and tp_pips > 0) else 0
            else:  # sell
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl_price = price + (sl_pips * symbol_info.point * 10) if (sl_pips and sl_pips > 0) else 0
                tp_price = price - (tp_pips * symbol_info.point * 10) if (tp_pips and tp_pips > 0) else 0
            
            # Get appropriate filling mode for this symbol
            filling_mode = self.get_filling_mode(symbol)
            
            # Prepare order request (โดยไม่ใส่ S/L และ T/P ถ้าไม่มี)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": trade_type,
                "price": price,
                "deviation": 20,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            # ✅ เพิ่ม S/L และ T/P เฉพาะเมื่อมีค่า (ป้องกัน error)
            if sl_price > 0:
                request["sl"] = sl_price
            if tp_price > 0:
                request["tp"] = tp_price
            
            # ⚠️ ตรวจสอบก่อนส่ง order
            print(f"\n📦 Order Details:")
            print(f"   Symbol: {symbol}")
            print(f"   Type: {order_type.upper()}")
            print(f"   Volume: {volume} lots")
            print(f"   Price: {price}")
            print(f"   S/L: {sl_price if sl_price > 0 else 'None'}")
            print(f"   T/P: {tp_price if tp_price > 0 else 'None'}")
            print(f"   Filling Mode: {filling_mode}")
            
            # ตรวจสอบ margin
            account_info = mt5.account_info()
            if account_info:
                print(f"   Balance: ${account_info.balance:.2f}")
                print(f"   Free Margin: ${account_info.margin_free:.2f}")
                
                # คำนวณ margin ที่ต้องใช้
                margin_needed = mt5.order_calc_margin(trade_type, symbol, volume, price)
                if margin_needed is not None:
                    print(f"   Margin Needed: ${margin_needed:.2f}")
                    if margin_needed > account_info.margin_free:
                        return {
                            'success': False,
                            'error': f'Insufficient margin: Need ${margin_needed:.2f}, Have ${account_info.margin_free:.2f}',
                            'retcode': 10019  # TRADE_RETCODE_NO_MONEY
                        }
            
            # ตรวจสอบ volume limits
            if volume < symbol_info.volume_min:
                return {
                    'success': False,
                    'error': f'Volume too small: {volume} < {symbol_info.volume_min} (min)',
                    'retcode': 10014
                }
            if volume > symbol_info.volume_max:
                return {
                    'success': False,
                    'error': f'Volume too large: {volume} > {symbol_info.volume_max} (max)',
                    'retcode': 10014
                }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                last_error = mt5.last_error()
                return {
                    'success': False, 
                    'error': f'Order send failed: {last_error}',
                    'retcode': 0
                }
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                # If filling mode failed, try alternative filling modes
                if result.retcode == 10030:  # TRADE_RETCODE_INVALID_FILL
                    # Try all filling modes
                    for fill_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                        if fill_mode == filling_mode:
                            continue  # Skip the one we already tried
                        
                        request["type_filling"] = fill_mode
                        result = mt5.order_send(request)
                        
                        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                            break
                
                # Check result again
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    error_msg = f'Order failed: Code {result.retcode if result else 0}'
                    if result:
                        error_msg += f' - {result.comment}'
                        print(f"\n❌ Order Failed Details:")
                        print(f"   Error Code: {result.retcode}")
                        print(f"   Comment: {result.comment}")
                        print(f"   Request ID: {result.request_id}")
                        if result.retcode == 10019:
                            print(f"   ⚠️  INSUFFICIENT MARGIN!")
                        elif result.retcode == 10014:
                            print(f"   ⚠️  INVALID VOLUME!")
                        elif result.retcode == 10004:
                            print(f"   ⚠️  MARKET CLOSED!")
                        elif result.retcode == 10030:
                            print(f"   ⚠️  INVALID FILLING MODE!")
                    
                    return {
                        'success': False, 
                        'error': error_msg,
                        'retcode': result.retcode if result else 0
                    }
            
            return {
                'success': True,
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'comment': result.comment
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_positions(self):
        """
        Get all open positions
        
        Returns:
            list: List of open positions
        """
        if not self.connected:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price': pos.price_open,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'comment': pos.comment,
                    'magic': pos.magic
                })
            
            return position_list
            
        except Exception as e:
            print(f"Error getting positions: {str(e)}")
            return []
    
    def close_position(self, ticket):
        """
        Close a specific position by ticket
        
        Args:
            ticket: Position ticket number
            
        Returns:
            dict: Close result
        """
        if not self.connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                return {'success': False, 'error': 'Position not found'}
            
            position = position[0]
            symbol = position.symbol
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': 'Failed to get current price'}
            
            # Determine close order type (opposite of open type)
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            # Get appropriate filling mode for this symbol
            filling_mode = self.get_filling_mode(symbol)
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": "Python close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result is None:
                return {'success': False, 'error': 'Close order failed'}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                # If filling mode failed, try alternative filling modes
                if result.retcode == 10030:  # TRADE_RETCODE_INVALID_FILL
                    for fill_mode in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
                        if fill_mode == filling_mode:
                            continue
                        
                        request["type_filling"] = fill_mode
                        result = mt5.order_send(request)
                        
                        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                            break
                
                # Check result again
                if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                    return {
                        'success': False,
                        'error': f'Close failed: {result.comment if result else "Unknown error"}'
                    }
            
            return {
                'success': True,
                'order': result.order,
                'comment': result.comment
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def close_all_positions(self):
        """
        Close all open positions
        
        Returns:
            int: Number of positions closed
        """
        if not self.connected:
            return 0
        
        positions = self.get_positions()
        closed_count = 0
        
        for position in positions:
            result = self.close_position(position['ticket'])
            if result['success']:
                closed_count += 1
                time.sleep(0.1)  # Small delay between closes
        
        return closed_count
    
    def get_symbol_positions(self, symbol):
        """
        Get positions for a specific symbol
        
        Args:
            symbol: Symbol name
            
        Returns:
            list: List of positions for the symbol
        """
        if not self.connected:
            return []
        
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'price': pos.price_open,
                    'profit': pos.profit
                })
            
            return position_list
            
        except Exception as e:
            print(f"Error getting symbol positions: {str(e)}")
            return []
    
    def modify_position(self, ticket, sl=None, tp=None):
        """
        Modify stop loss and take profit of an existing position
        
        Args:
            ticket: Position ticket
            sl: New stop loss price (None to keep current)
            tp: New take profit price (None to keep current)
            
        Returns:
            dict: Modification result
        """
        if not self.connected:
            return {'success': False, 'error': 'Not connected to MT5'}
        
        try:
            # Get position
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                return {'success': False, 'error': 'Position not found'}
            
            position = position[0]
            
            # Use current values if not specified
            new_sl = sl if sl is not None else position.sl
            new_tp = tp if tp is not None else position.tp
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "sl": new_sl,
                "tp": new_tp,
                "position": ticket
            }
            
            # Send modification
            result = mt5.order_send(request)
            
            if result is None:
                return {'success': False, 'error': 'Modification failed'}
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f'Modification failed: {result.comment}'
                }
            
            return {
                'success': True,
                'comment': result.comment
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_account_history(self, days=7):
        """
        Get account trading history
        
        Args:
            days: Number of days to look back
            
        Returns:
            list: List of historical deals
        """
        if not self.connected:
            return []
        
        try:
            from_date = datetime.now() - timedelta(days=days)
            to_date = datetime.now()
            
            deals = mt5.history_deals_get(from_date, to_date)
            if deals is None or len(deals) == 0:
                return []
            
            history_list = []
            for deal in deals:
                history_list.append({
                    'ticket': deal.ticket,
                    'order': deal.order,
                    'symbol': deal.symbol,
                    'type': 'BUY' if deal.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'time': datetime.fromtimestamp(deal.time),
                    'comment': deal.comment
                })
            
            return history_list
            
        except Exception as e:
            print(f"Error getting history: {str(e)}")
            return []
