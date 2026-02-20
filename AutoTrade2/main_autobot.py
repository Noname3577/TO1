"""
MetaTrader 5 AI Trading Bot with Advanced Analysis
Main Application with Auto Trading Features
Created: 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import MetaTrader5 as mt5
from datetime import datetime
import threading
import json
import sys

from mt5_trader import MT5Trader
from trading_bot import TradingBot

class ConsoleRedirector:
    def __init__(self, text_widget, tag='info', max_lines=500):
        self.text_widget = text_widget
        self.tag = tag
        self.max_lines = max_lines  
        self.line_count = 0
        
    def write(self, message):
        if message.strip():  
            self.line_count += message.count('\n')
            
            
            if self.line_count > self.max_lines:
                self.clear_old_lines()
            
            self.text_widget.insert('end', message, self.tag)
            self.text_widget.see('end')  
            self.text_widget.update_idletasks()
    
    def clear_old_lines(self):
        """ลบบรรทัดเก่าเหลือแค่ครึ่งระบบ"""
        try:
            # ลบครึ่งแรกออก (เก็บไว้ครึ่งหลัง)
            lines_to_keep = self.max_lines // 2
            content = self.text_widget.get('1.0', 'end')
            lines = content.split('\n')
            
            if len(lines) > self.max_lines:
                # เก็บเฉพาะบรรทัดหลัง
                new_content = '\n'.join(lines[-lines_to_keep:])
                self.text_widget.delete('1.0', 'end')
                self.text_widget.insert('1.0', f"[... cleared {len(lines) - lines_to_keep} old lines ...]\n\n" + new_content)
                self.line_count = lines_to_keep
        except Exception as e:
            pass  # Ignore errors during clearing
    
    def flush(self):
        pass

class AITradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 บอทเทรดอัตโนมัติ AI - MT5 AutoTrader 2026")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0a0e27')
        
        # Initialize components
        self.trader = MT5Trader()
        self.bot = None
        self.is_connected = False
        self.auto_trading_active = False
        self.bot_thread = None

        # Hidden text widgets (ใช้สำหรับ log() และ analyze_manual() — ไม่แสดงใน UI)
        self.log_text = scrolledtext.ScrolledText(self.root)
        self.analysis_text = scrolledtext.ScrolledText(self.root)
        for w in (self.log_text, self.analysis_text):
            for tag, clr in [('success','#00ff88'),('error','#ff4757'),
                             ('info','#00d4ff'),('warning','#ffa502'),
                             ('title','#00d4ff'),('danger','#ff4757')]:
                w.tag_config(tag, foreground=clr)
        
        # Configure styles
        self.setup_styles()
        
        # Create UI
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
        # Auto-connect
        self.root.after(500, self.connect_mt5)
        
    def setup_styles(self):
        """Setup modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors
        bg_dark = '#0a0e27'
        bg_medium = '#151b3d'
        bg_light = '#1e2749'
        accent = '#00d4ff'
        success = '#00ff88'
        danger = '#ff4757'
        warning = '#ffa502'
        
        # Button styles
        style.configure('Accent.TButton', background=accent, foreground='#000000',
                       borderwidth=0, padding=10, font=('Segoe UI', 10, 'bold'))
        style.configure('Success.TButton', background=success, foreground='#000000',
                       borderwidth=0, padding=10, font=('Segoe UI', 10, 'bold'))
        style.configure('Danger.TButton', background=danger, foreground='#ffffff',
                       borderwidth=0, padding=10, font=('Segoe UI', 10, 'bold'))
        
        # Combobox style
        style.configure('TCombobox',
                       fieldbackground='#1e2749',
                       background='#1e2749',
                       foreground='#ffffff',
                       arrowcolor='#00d4ff',
                       borderwidth=0,
                       relief='flat')
        style.map('TCombobox',
                 fieldbackground=[('readonly', '#1e2749')],
                 selectbackground=[('readonly', '#1e2749')],
                 selectforeground=[('readonly', '#ffffff')])
        
    def create_header(self):
        """Create header"""
        header = tk.Frame(self.root, bg='#0a0e27', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🤖 บอทเทรดอัตโนมัติ AI - ระบบวิเคราะห์ขั้นสูง",
                        font=('Segoe UI', 22, 'bold'), bg='#0a0e27', fg='#00d4ff')
        title.pack(side='left', padx=30, pady=20)
        
        self.connection_label = tk.Label(header, text="● ไม่ได้เชื่อมต่อ",
                                        font=('Segoe UI', 11), bg='#0a0e27', fg='#ff4757')
        self.connection_label.pack(side='right', padx=30)
        
        self.connect_btn = tk.Button(header, text="🔌 เชื่อมต่อ MT5",
                                     font=('Segoe UI', 10, 'bold'),
                                     bg='#00d4ff', fg='#000000',
                                     relief='flat', cursor='hand2',
                                     command=self.connect_mt5)
        self.connect_btn.pack(side='right', padx=10)
        
    def create_main_content(self):
        """Create main layout: left nav sidebar + right content pages"""
        body = tk.Frame(self.root, bg='#0a0e27')
        body.pack(fill='both', expand=True)

        # ── Left nav sidebar ──────────────────────────────────────────────
        nav = tk.Frame(body, bg='#0d1130', width=88)
        nav.pack(side='left', fill='y')
        nav.pack_propagate(False)
        self.create_nav_sidebar(nav)

        # 1-px vertical divider
        tk.Frame(body, bg='#1e2749', width=1).pack(side='left', fill='y')

        # ── Content area ──────────────────────────────────────────────────
        content = tk.Frame(body, bg='#0a0e27')
        content.pack(side='left', fill='both', expand=True)

        # Three page frames
        self.page_dashboard = tk.Frame(content, bg='#0a0e27')
        self.page_console   = tk.Frame(content, bg='#0a0e27')
        self.page_control   = tk.Frame(content, bg='#151b3d')

        self.create_dashboard_panel(self.page_dashboard)
        self.create_console_panel(self.page_console)
        self.create_bot_control(self.page_control)

        # Show dashboard (home) first
        self._current_page = None
        self.show_page('dashboard')

    def create_nav_sidebar(self, parent):
        """Create vertical nav sidebar with icon buttons"""
        # ── Top logo ──────────────────────────────────────────────────────
        tk.Label(parent, text='⚡', font=('Segoe UI', 22, 'bold'),
                 bg='#0d1130', fg='#00d4ff').pack(pady=(16, 4))
        tk.Frame(parent, bg='#1e2749', height=1).pack(fill='x', padx=10, pady=(0, 10))

        # ── Nav items ────────────────────────────────────────────────────
        nav_items = [
            ('dashboard', '🏠', 'หน้าแรก'),
            ('console',   '💻', 'Console'),
            ('control',   '🤖', 'ควบคุมบอท'),
        ]
        self._nav_btns = {}

        for page_id, icon, label in nav_items:
            outer = tk.Frame(parent, bg='#0d1130', cursor='hand2')
            outer.pack(fill='x', pady=2)

            icon_lbl = tk.Label(outer, text=icon, font=('Segoe UI', 20),
                                bg='#0d1130', fg='#4a5080')
            icon_lbl.pack(pady=(10, 0))

            text_lbl = tk.Label(outer, text=label, font=('Segoe UI', 7),
                                bg='#0d1130', fg='#4a5080',
                                wraplength=80, justify='center')
            text_lbl.pack(pady=(2, 10))

            # Capture loop vars with default args
            def _click(e, p=page_id):               self.show_page(p)
            def _enter(e, o=outer, il=icon_lbl, tl=text_lbl, p=page_id):
                if self._current_page != p:
                    for w in (o, il, tl): w.config(bg='#151b3d')
                    il.config(fg='#c0c8ff'); tl.config(fg='#c0c8ff')
            def _leave(e, p=page_id):               self._refresh_nav_btn(p)

            for w in (outer, icon_lbl, text_lbl):
                w.bind('<Button-1>', _click)
                w.bind('<Enter>',    _enter)
                w.bind('<Leave>',    _leave)

            self._nav_btns[page_id] = (outer, icon_lbl, text_lbl)

    def show_page(self, page_name):
        """Switch the visible page and update nav highlight"""
        page_map = {
            'dashboard': self.page_dashboard,
            'console':   self.page_console,
            'control':   self.page_control,
        }
        if page_name not in page_map:
            return
        for p in page_map.values():
            p.pack_forget()
        page_map[page_name].pack(fill='both', expand=True)
        self._current_page = page_name
        for pid in self._nav_btns:
            self._refresh_nav_btn(pid)

    def _refresh_nav_btn(self, page_id):
        """Apply active / inactive style to a nav button"""
        outer, icon_lbl, text_lbl = self._nav_btns[page_id]
        is_active = (page_id == self._current_page)
        if is_active:
            bg, fg = '#151b3d', '#00d4ff'
        else:
            bg, fg = '#0d1130', '#4a5080'
        for w in (outer, icon_lbl, text_lbl):
            w.config(bg=bg)
        icon_lbl.config(fg=fg)
        text_lbl.config(fg=fg)

    def create_dashboard_panel(self, parent):
        """สร้าง Dashboard หน้าแสดงผลภาพรวมแบบเข้าใจง่าย"""
        # ── Scroll canvas ────────────────────────────────────
        canvas = tk.Canvas(parent, bg='#0a0e27', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        self.dash_frame = tk.Frame(canvas, bg='#0a0e27')
        _dash_win = canvas.create_window((0, 0), window=self.dash_frame, anchor='nw')

        # Resize inner frame width when canvas changes size
        def _on_canvas_resize(e):
            canvas.itemconfig(_dash_win, width=e.width)
        canvas.bind('<Configure>', _on_canvas_resize)
        self.dash_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        # Scroll wheel helper — bind recursively to every child
        def _scroll(e):
            canvas.yview_scroll(int(-1 * (e.delta // 120)), 'units')
        def _bind_scroll(widget):
            widget.bind('<MouseWheel>', _scroll)
            for child in widget.winfo_children():
                _bind_scroll(child)
        canvas.bind('<MouseWheel>', _scroll)
        # Re-bind after widgets are built (called at end)
        self._dash_canvas = canvas
        self._dash_bind_scroll = _bind_scroll

        # ── Row 0: Stat Cards ─────────────────────────────────
        cards_frame = tk.Frame(self.dash_frame, bg='#0a0e27')
        cards_frame.pack(fill='x', padx=10, pady=(10, 6))
        cards_frame.columnconfigure((0,1,2,3,4), weight=1)

        def stat_card(col, title, var_name, unit='', color='#00d4ff'):
            f = tk.Frame(cards_frame, bg='#151b3d')
            f.grid(row=0, column=col, sticky='nsew', padx=4)
            tk.Label(f, text=title, font=('Segoe UI', 8),
                     bg='#151b3d', fg='#7a8099').pack(pady=(8, 0))
            lbl = tk.Label(f, text='──', font=('Segoe UI', 17, 'bold'),
                           bg='#151b3d', fg=color)
            lbl.pack(pady=(2, 0))
            tk.Label(f, text=unit, font=('Segoe UI', 8),
                     bg='#151b3d', fg='#7a8099').pack(pady=(0, 8))
            setattr(self, var_name, lbl)

        stat_card(0, '💰 ยอดเงิน',            'dash_balance_lbl',   'USD',     '#00ff88')
        stat_card(1, '📈 กำไร/ขาดทุนวันนี้',   'dash_daily_pl_lbl',  'USD',     '#ffa502')
        stat_card(2, '📂 ออเดอร์ที่เปิด',       'dash_open_pos_lbl',  'ออเดอร์', '#00d4ff')
        stat_card(3, '🏆 Win Rate',             'dash_winrate_lbl',   '%',       '#a29bfe')
        stat_card(4, '⚡ สถานะบอท',             'dash_bot_status_lbl','',        '#00ff88')

        # ── Circuit Breaker Banner (hidden by default) ─────────
        self.dash_breaker_frame = tk.Frame(self.dash_frame, bg='#3d0000')
        self.dash_breaker_lbl = tk.Label(self.dash_breaker_frame,
            text='🛑  CIRCUIT BREAKER ACTIVE',
            font=('Segoe UI', 11, 'bold'), bg='#3d0000', fg='#ff4757')
        self.dash_breaker_reason = tk.Label(self.dash_breaker_frame,
            text='', font=('Segoe UI', 9), bg='#3d0000', fg='#ffa502')
        # Pack now to reserve position, then hide
        self.dash_breaker_frame.pack(fill='x', padx=10, pady=(0, 2))
        self.dash_breaker_frame.pack_forget()

        # ── Row 1: Two-column: Positions (left) + Risk (right) ─
        row1 = tk.Frame(self.dash_frame, bg='#0a0e27')
        row1.pack(fill='x', padx=10, pady=4)
        row1.columnconfigure(0, weight=3)
        row1.columnconfigure(1, weight=2)

        # --- Left: Open Positions Table
        pos_frame = tk.Frame(row1, bg='#151b3d')
        pos_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 4))

        tk.Label(pos_frame, text='📂  ออเดอร์ที่เปิดอยู่',
                 font=('Segoe UI', 10, 'bold'), bg='#151b3d',
                 fg='#00d4ff').pack(anchor='w', padx=10, pady=(8, 2))

        cols = ('ticket', 'symbol', 'type', 'lots', 'open_price', 'current', 'profit', 'time')
        col_names = ('Ticket', 'Symbol', 'ประเภท', 'Lots', 'ราคาเปิด', 'ราคาปัจจุบัน', 'กำไร ($)', 'เวลา')
        col_widths = (68, 72, 52, 50, 88, 98, 88, 80)

        style = ttk.Style()
        style.theme_use('default')
        style.configure('Dash.Treeview', background='#1e2749', foreground='#ffffff',
                        fieldbackground='#1e2749', rowheight=24, font=('Segoe UI', 9))
        style.configure('Dash.Treeview.Heading', background='#0a0e27',
                        foreground='#00d4ff', font=('Segoe UI', 8, 'bold'), relief='flat')
        style.map('Dash.Treeview', background=[('selected', '#2d3a6b')])

        tree_wrap = tk.Frame(pos_frame, bg='#151b3d')
        tree_wrap.pack(fill='x', padx=8, pady=(0, 8))

        self.pos_tree = ttk.Treeview(tree_wrap, columns=cols, show='headings',
                                     height=5, style='Dash.Treeview')
        for c, nm, w in zip(cols, col_names, col_widths):
            self.pos_tree.heading(c, text=nm)
            self.pos_tree.column(c, width=w, anchor='center', stretch=True)
        self.pos_tree.tag_configure('buy',  foreground='#00ff88')
        self.pos_tree.tag_configure('sell', foreground='#ff6b7a')
        self.pos_tree.tag_configure('loss', foreground='#ff4757')

        vsb_tree = ttk.Scrollbar(tree_wrap, orient='vertical', command=self.pos_tree.yview)
        self.pos_tree.configure(yscrollcommand=vsb_tree.set)
        self.pos_tree.pack(side='left', fill='both', expand=True)
        vsb_tree.pack(side='right', fill='y')

        # --- Right: Risk Protection
        risk_frame = tk.Frame(row1, bg='#151b3d')
        risk_frame.grid(row=0, column=1, sticky='nsew', padx=(4, 0))

        tk.Label(risk_frame, text='🛡️  ระบบป้องกันความเสี่ยง',
                 font=('Segoe UI', 10, 'bold'), bg='#151b3d',
                 fg='#00d4ff').pack(anchor='w', padx=10, pady=(8, 4))

        risk_inner = tk.Frame(risk_frame, bg='#151b3d')
        risk_inner.pack(fill='x', padx=10, pady=(0, 10))

        def risk_row(parent, row, label, var_name_val, var_name_bar, var_name_status):
            tk.Label(parent, text=label, font=('Segoe UI', 8), bg='#151b3d',
                     fg='#b0b0b0', anchor='w').grid(
                     row=row*2, column=0, columnspan=2, sticky='w', pady=(6, 0))
            val_lbl = tk.Label(parent, text='─', font=('Segoe UI', 8, 'bold'),
                               bg='#151b3d', fg='#ffffff', anchor='w')
            val_lbl.grid(row=row*2, column=2, sticky='e', padx=(4, 0), pady=(6, 0))
            status_lbl = tk.Label(parent, text='✅', font=('Segoe UI', 9),
                                  bg='#151b3d', fg='#00ff88')
            status_lbl.grid(row=row*2, column=3, padx=(4, 0), pady=(6, 0))
            bar_canvas = tk.Canvas(parent, height=10, bg='#0a0e27',
                                   highlightthickness=0)
            bar_canvas.grid(row=row*2+1, column=0, columnspan=4,
                            sticky='ew', pady=(1, 2))
            setattr(self, var_name_val, val_lbl)
            setattr(self, var_name_bar, bar_canvas)
            setattr(self, var_name_status, status_lbl)
            parent.columnconfigure(0, weight=1)

        risk_row(risk_inner, 0, '📅 ขาดทุน/วัน',     'dash_daily_loss_val', 'dash_daily_loss_bar', 'dash_daily_loss_st')
        risk_row(risk_inner, 1, '📉 Drawdown',         'dash_drawdown_val',   'dash_drawdown_bar',   'dash_drawdown_st')
        risk_row(risk_inner, 2, '🔢 เทรด/วัน',        'dash_trades_val',     'dash_trades_bar',     'dash_trades_st')
        risk_row(risk_inner, 3, '❌ แพ้ติดกัน',        'dash_consec_val',     'dash_consec_bar',     'dash_consec_st')

        # ── Row 2: Recent Activity ────────────────────────────
        act_frame = tk.Frame(self.dash_frame, bg='#151b3d')
        act_frame.pack(fill='x', padx=10, pady=4)

        act_hdr = tk.Frame(act_frame, bg='#151b3d')
        act_hdr.pack(fill='x', padx=10, pady=(6, 2))
        tk.Label(act_hdr, text='📋  กิจกรรมล่าสุด',
                 font=('Segoe UI', 10, 'bold'), bg='#151b3d',
                 fg='#00d4ff').pack(side='left')
        tk.Button(act_hdr, text='🔄 รีเฟรช', font=('Segoe UI', 8),
                  bg='#1e2749', fg='#00d4ff', relief='flat', cursor='hand2',
                  bd=0, padx=8, pady=3,
                  command=self.update_dashboard).pack(side='right')
        self.dash_updated_lbl = tk.Label(act_hdr, text='', font=('Segoe UI', 8),
                                          bg='#151b3d', fg='#555555')
        self.dash_updated_lbl.pack(side='right', padx=8)

        self.dash_activity = tk.Frame(act_frame, bg='#0d1130')
        self.dash_activity.pack(fill='x', padx=10, pady=(0, 8))

        self.dash_activity_labels = []
        for i in range(10):
            bg = '#0d1130' if i % 2 == 0 else '#111630'
            lbl = tk.Label(self.dash_activity, text='', font=('Consolas', 8),
                           bg=bg, fg='#7a8099', anchor='w', padx=8, pady=3)
            lbl.pack(fill='x')
            self.dash_activity_labels.append(lbl)

        # Internal log buffer for activity feed
        self._dash_log_buffer = []

        # Bind scroll wheel to everything after build
        self.dash_frame.after(100, lambda: _bind_scroll(self.dash_frame))

        # Start auto-refresh
        self.update_dashboard()

    def _draw_risk_bar(self, canvas, percent, max_percent, warn_pct=60, danger_pct=85):
        """วาด progress bar สำหรับ risk (fluid width)"""
        canvas.update_idletasks()
        w = max(canvas.winfo_width(), 40)
        h = 10
        canvas.config(height=h)
        canvas.delete('all')
        ratio = min(percent / max_percent, 1.0) if max_percent > 0 else 0
        fill_w = int(w * ratio)
        pct_of_max = ratio * 100
        if pct_of_max < warn_pct:
            color = '#00ff88'
        elif pct_of_max < danger_pct:
            color = '#ffa502'
        else:
            color = '#ff4757'
        canvas.create_rectangle(0, 0, w, h, fill='#1a1f3d', outline='')
        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline='')

    def update_dashboard(self):
        """อัปเดตข้อมูลทุก widget ใน Dashboard"""
        try:
            rm = self.bot.risk_manager if self.bot else None

            # ── Stat Cards ────────────────────────────────────
            if self.is_connected and self.trader:
                acc = self.trader.get_account_info()
                if acc:
                    bal = acc.get('balance', 0)
                    equity = acc.get('equity', bal)
                    self.dash_balance_lbl.config(text=f"${bal:,.2f}", fg='#00ff88')

            if rm:
                daily_pl = rm.daily_loss * -1 if rm.daily_loss < 0 else rm.daily_loss
                sign = '+' if rm.daily_loss >= 0 else ''
                raw = rm.account_balance - rm.daily_start_balance
                color_pl = '#00ff88' if raw >= 0 else '#ff4757'
                self.dash_daily_pl_lbl.config(
                    text=f"{'+' if raw >= 0 else ''}{raw:.2f}",
                    fg=color_pl)

                # Win rate
                if rm.trade_history:
                    wins = sum(1 for t in rm.trade_history if t.get('was_win'))
                    wr = wins / len(rm.trade_history) * 100
                    wr_color = '#00ff88' if wr >= 55 else '#ffa502' if wr >= 40 else '#ff4757'
                    self.dash_winrate_lbl.config(text=f"{wr:.1f}", fg=wr_color)
                else:
                    self.dash_winrate_lbl.config(text="─", fg='#555555')

                # Circuit breaker
                cb = rm.check_circuit_breaker()
                if cb['active']:
                    self.dash_breaker_frame.pack(fill='x', padx=10, pady=(0, 2))
                    self.dash_breaker_lbl.pack(pady=(6,0))
                    self.dash_breaker_reason.config(text=f"สาเหตุ: {cb['reason']}")
                    self.dash_breaker_reason.pack(pady=(0,6))
                else:
                    self.dash_breaker_frame.pack_forget()

            # ── Bot Status ────────────────────────────────────
            if self.auto_trading_active if hasattr(self, 'auto_trading_active') else False:
                self.dash_bot_status_lbl.config(text='🟢 ทำงาน', fg='#00ff88')
            elif self.is_connected:
                self.dash_bot_status_lbl.config(text='🟡 เชื่อมต่อแล้ว', fg='#ffa502')
            else:
                self.dash_bot_status_lbl.config(text='🔴 ไม่เชื่อมต่อ', fg='#ff4757')

            # ── Positions Table ───────────────────────────────
            for row in self.pos_tree.get_children():
                self.pos_tree.delete(row)

            if self.is_connected:
                positions = mt5.positions_get()
                count = len(positions) if positions else 0
                self.dash_open_pos_lbl.config(
                    text=str(count),
                    fg='#ffa502' if count > 0 else '#a0a0a0')

                if positions:
                    for pos in positions:
                        tick = mt5.symbol_info_tick(pos.symbol)
                        cur_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask if tick else 0
                        ptype = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'
                        tag = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
                        if pos.profit < 0:
                            tag = 'loss'
                        profit_str = f"{'+'if pos.profit>=0 else ''}{pos.profit:.2f}"
                        open_dt = datetime.fromtimestamp(pos.time).strftime('%H:%M:%S')
                        self.pos_tree.insert('', 'end', tags=(tag,), values=(
                            pos.ticket,
                            pos.symbol,
                            ptype,
                            f"{pos.volume:.2f}",
                            f"{pos.price_open:.5f}",
                            f"{cur_price:.5f}",
                            profit_str,
                            open_dt,
                        ))
            else:
                self.dash_open_pos_lbl.config(text='─', fg='#555555')

            # ── Risk Bars ─────────────────────────────────────
            if rm:
                # Daily loss
                dl_pct = abs(rm.daily_loss / rm.daily_start_balance * 100) if rm.daily_start_balance else 0
                self.dash_daily_loss_val.config(
                    text=f"{dl_pct:.1f}% / {rm.daily_loss_limit_percent}%",
                    fg='#ff4757' if dl_pct >= rm.daily_loss_limit_percent * 0.85 else '#ffffff')
                self._draw_risk_bar(self.dash_daily_loss_bar, dl_pct, rm.daily_loss_limit_percent)
                self.dash_daily_loss_st.config(
                    text='🔴' if dl_pct >= rm.daily_loss_limit_percent else ('🟡' if dl_pct >= rm.daily_loss_limit_percent * 0.6 else '✅'),
                    fg='#ff4757' if dl_pct >= rm.daily_loss_limit_percent else '#ffa502' if dl_pct >= rm.daily_loss_limit_percent*0.6 else '#00ff88')

                # Drawdown
                dd = rm.current_drawdown_percent
                self.dash_drawdown_val.config(
                    text=f"{dd:.2f}% / {rm.max_drawdown_percent}%",
                    fg='#ff4757' if dd >= rm.max_drawdown_percent * 0.85 else '#ffffff')
                self._draw_risk_bar(self.dash_drawdown_bar, dd, rm.max_drawdown_percent)
                self.dash_drawdown_st.config(
                    text='🔴' if dd >= rm.max_drawdown_percent else ('🟡' if dd >= rm.max_drawdown_percent * 0.6 else '✅'),
                    fg='#ff4757' if dd >= rm.max_drawdown_percent else '#ffa502' if dd >= rm.max_drawdown_percent*0.6 else '#00ff88')

                # Daily trades
                if rm.max_daily_trades > 0:
                    self.dash_trades_val.config(
                        text=f"{rm.daily_trade_count} / {rm.max_daily_trades}",
                        fg='#ff4757' if rm.daily_trade_count >= rm.max_daily_trades else '#ffffff')
                    self._draw_risk_bar(self.dash_trades_bar, rm.daily_trade_count, rm.max_daily_trades)
                    self.dash_trades_st.config(
                        text='🔴' if rm.daily_trade_count >= rm.max_daily_trades else '✅',
                        fg='#ff4757' if rm.daily_trade_count >= rm.max_daily_trades else '#00ff88')
                else:
                    self.dash_trades_val.config(text=f"{rm.daily_trade_count} / ∞", fg='#a0a0a0')
                    self._draw_risk_bar(self.dash_trades_bar, 0, 1)
                    self.dash_trades_st.config(text='✅', fg='#00ff88')

                # Consecutive losses
                cl = rm.consecutive_losses
                self.dash_consec_val.config(
                    text=f"{cl} / {rm.max_consecutive_losses}",
                    fg='#ff4757' if cl >= rm.max_consecutive_losses else '#ffffff')
                self._draw_risk_bar(self.dash_consec_bar, cl, rm.max_consecutive_losses)
                self.dash_consec_st.config(
                    text='🔴' if cl >= rm.max_consecutive_losses else ('🟡' if cl >= rm.max_consecutive_losses - 1 else '✅'),
                    fg='#ff4757' if cl >= rm.max_consecutive_losses else '#ffa502' if cl >= rm.max_consecutive_losses - 1 else '#00ff88')

            # ── Timestamp ─────────────────────────────────────
            now = datetime.now().strftime('%H:%M:%S')
            self.dash_updated_lbl.config(text=f'อัปเดตล่าสุด: {now}')

        except Exception as e:
            pass  # ไม่ให้ crash ถ้า MT5 ยังไม่เชื่อมต่อ

        # Auto-refresh ทุก 5 วินาที
        self.root.after(5000, self.update_dashboard)

    def dash_add_activity(self, text, color='#a0a0a0'):
        """เพิ่ม event เข้า activity feed ใน Dashboard"""
        ts = datetime.now().strftime('%H:%M:%S')
        self._dash_log_buffer.insert(0, (f"[{ts}] {text}", color))
        self._dash_log_buffer = self._dash_log_buffer[:10]
        for i, (msg, clr) in enumerate(self._dash_log_buffer):
            self.dash_activity_labels[i].config(text=msg, fg=clr)

    def create_bot_control(self, parent):
        """Create bot control panel (scrollable full page)"""
        ctrl_canvas = tk.Canvas(parent, bg='#151b3d', highlightthickness=0)
        ctrl_vsb = ttk.Scrollbar(parent, orient='vertical', command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_vsb.set)
        ctrl_vsb.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)

        def _ctrl_scroll(e):
            ctrl_canvas.yview_scroll(int(-1 * (e.delta // 120)), 'units')
        def _bind_ctrl_scroll(widget):
            widget.bind('<MouseWheel>', _ctrl_scroll)
            for child in widget.winfo_children():
                _bind_ctrl_scroll(child)
        ctrl_canvas.bind('<MouseWheel>', _ctrl_scroll)

        card = tk.Frame(ctrl_canvas, bg='#151b3d', relief='flat')
        win_id = ctrl_canvas.create_window((0, 0), window=card, anchor='nw')
        card.bind('<Configure>', lambda e: ctrl_canvas.configure(
            scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind('<Configure>', lambda e: ctrl_canvas.itemconfig(win_id, width=e.width))
        
        header = tk.Label(card, text="🤖 ควบคุมบอทเทรดอัตโนมัติ",
                         font=('Segoe UI', 14, 'bold'), bg='#151b3d', fg='#00d4ff')
        header.pack(anchor='w', padx=20, pady=(10, 15))
        
        # Settings
        settings_frame = tk.Frame(card, bg='#151b3d')
        settings_frame.pack(fill='x', padx=20, pady=10)
        
        # Symbol
        tk.Label(settings_frame, text="คู่เงิน:", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=0, column=0, sticky='w', pady=8)
        self.bot_symbol_var = tk.StringVar(value='EURUSD')
        
        # Popular currency pairs
        symbol_choices = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
            'USDCHF', 'USDCAD', 'NZDUSD', 'EURGBP',
            'EURJPY', 'GBPJPY', 'AUDJPY', 'XAUUSD'
        ]
        
        symbol_combo = ttk.Combobox(settings_frame, textvariable=self.bot_symbol_var,
                                    values=symbol_choices,
                                    font=('Segoe UI', 10),
                                    width=13,
                                    state='readonly')
        symbol_combo.grid(row=0, column=1, sticky='w', pady=8, padx=(10, 0))
        
        # Risk %
        tk.Label(settings_frame, text="เสี่ยง %:", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=1, column=0, sticky='w', pady=8)
        self.bot_risk_var = tk.StringVar(value='2.0')
        tk.Entry(settings_frame, textvariable=self.bot_risk_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=1, column=1, sticky='w', pady=8, padx=(10, 0))
        
        # Fixed Volume (Lots)
        tk.Label(settings_frame, text="📊 Volume (Lots):", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=2, column=0, sticky='w', pady=8)
        self.fixed_volume_var = tk.StringVar(value='0.01')
        tk.Entry(settings_frame, textvariable=self.fixed_volume_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=2, column=1, sticky='w', pady=8, padx=(10, 0))
        
        # 💡 คำอธิบาย Volume
        tk.Label(settings_frame, text="💡 ขนาด Lot ที่จะสั่งทุกครั้ง (เช่น 0.01, 0.1, 1.0)",
                font=('Segoe UI', 8, 'italic'), bg='#151b3d', fg='#00d4ff').grid(
                row=2, column=2, columnspan=2, sticky='w', pady=8, padx=(30, 0))
        
        # Min Quality Score
        tk.Label(settings_frame, text="คุณภาพขั้นต่ำ:", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=0, column=2, sticky='w', pady=8, padx=(30, 0))
        self.min_quality_var = tk.StringVar(value='60')
        tk.Entry(settings_frame, textvariable=self.min_quality_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=0, column=3, sticky='w', pady=8, padx=(10, 0))
        
        # Scan Interval
        tk.Label(settings_frame, text="สแกน (วินาที):", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=1, column=2, sticky='w', pady=8, padx=(30, 0))
        self.scan_interval_var = tk.StringVar(value='60')
        tk.Entry(settings_frame, textvariable=self.scan_interval_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=1, column=3, sticky='w', pady=8, padx=(10, 0))
        
        # �️ การตั้งค่าระบบป้องกัน
        protection_label = tk.Label(card, text="🛡️ การป้องกันความเสี่ยง",
                                    font=('Segoe UI', 11, 'bold'),
                                    bg='#151b3d', fg='#00d4ff')
        protection_label.pack(anchor='w', padx=20, pady=(15, 5))
        
        protection_frame = tk.Frame(card, bg='#151b3d')
        protection_frame.pack(fill='x', padx=20, pady=5)
        
        # Max Daily Trades
        tk.Label(protection_frame, text="จำกัดเทรด/วัน:", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=0, column=0, sticky='w', pady=5)
        self.max_daily_trades_var = tk.StringVar(value='20')
        tk.Entry(protection_frame, textvariable=self.max_daily_trades_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=0, column=1, sticky='w', pady=5, padx=(10, 0))
        
        # 💡 คำอธิบาย - ใส่ 0 = ไม่จำกัด
        tk.Label(protection_frame, text="💡 ใส่ 0 = ไม่จำกัดครั้ง",
                font=('Segoe UI', 8, 'italic'), bg='#151b3d', fg='#ffa502').grid(
                row=1, column=0, columnspan=2, sticky='w', pady=(0, 5))
        
        # Daily Loss Limit
        tk.Label(protection_frame, text="จำกัดขาดทุน/วัน (%):", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=0, column=2, sticky='w', pady=5, padx=(30, 0))
        self.daily_loss_limit_var = tk.StringVar(value='5.0')
        tk.Entry(protection_frame, textvariable=self.daily_loss_limit_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=0, column=3, sticky='w', pady=5, padx=(10, 0))
        
        # Max Drawdown
        tk.Label(protection_frame, text="Drawdown สูงสุด (%):", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=2, column=0, sticky='w', pady=5)
        self.max_drawdown_var = tk.StringVar(value='15.0')
        tk.Entry(protection_frame, textvariable=self.max_drawdown_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=2, column=1, sticky='w', pady=5, padx=(10, 0))
        
        # Max Consecutive Losses
        tk.Label(protection_frame, text="จำกัดเสียติดกัน:", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=2, column=2, sticky='w', pady=5, padx=(30, 0))
        self.max_consecutive_losses_var = tk.StringVar(value='3')
        tk.Entry(protection_frame, textvariable=self.max_consecutive_losses_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=2, column=3, sticky='w', pady=5, padx=(10, 0))
        
        # Max Loss Per Trade (USD)
        tk.Label(protection_frame, text="จำกัดขาดทุนต่อออเดอร์ ($):", font=('Segoe UI', 10),
                bg='#151b3d', fg='#ffffff').grid(row=3, column=0, sticky='w', pady=5)
        self.max_loss_per_trade_var = tk.StringVar(value='0')
        tk.Entry(protection_frame, textvariable=self.max_loss_per_trade_var,
                font=('Segoe UI', 10), bg='#1e2749', fg='#ffffff',
                insertbackground='#ffffff', relief='flat', width=15).grid(
                row=3, column=1, sticky='w', pady=5, padx=(10, 0))
        
        # 💡 คำอธิบาย - ใส่ 0 = ไม่จำกัด
        tk.Label(protection_frame, text="💡 ใส่ 0 = ไม่จำกัด, เช่น 10 = ปิดเมื่อขาดทุน $10",
                font=('Segoe UI', 8, 'italic'), bg='#151b3d', fg='#ffa502').grid(
                row=4, column=0, columnspan=4, sticky='w', pady=(0, 5))
        
        # คำอธิบาย
        hint_label = tk.Label(card, 
                text="💡 ปรับค่าเหล่านี้เพื่อควบคุมความเสี่ยง - ค่าสูง = เทรดได้มากขึ้น แต่เสี่ยงมากขึ้น",
                font=('Segoe UI', 8, 'italic'), bg='#151b3d', fg='#a0a0a0')
        hint_label.pack(anchor='w', padx=20, pady=(0, 10))
        
        # Control buttons
        btn_frame = tk.Frame(card, bg='#151b3d')
        btn_frame.pack(fill='x', padx=20, pady=15)
        
        self.start_bot_btn = tk.Button(btn_frame, text="▶️ เริ่มเทรดอัตโนมัติ",
                                       font=('Segoe UI', 12, 'bold'),
                                       bg='#00ff88', fg='#000000',
                                       activebackground='#00cc66',
                                       relief='flat', cursor='hand2',
                                       command=self.start_auto_trading)
        self.start_bot_btn.pack(side='left', fill='x', expand=True, padx=(0, 5), ipady=10)
        
        self.stop_bot_btn = tk.Button(btn_frame, text="⏹️ หยุดบอท",
                                      font=('Segoe UI', 12, 'bold'),
                                      bg='#ff4757', fg='#ffffff',
                                      activebackground='#cc3845',
                                      relief='flat', cursor='hand2',
                                      state='disabled',
                                      command=self.stop_auto_trading)
        self.stop_bot_btn.pack(side='left', fill='x', expand=True, padx=(5, 0), ipady=10)
        
        # Manual analyze button
        analyze_btn = tk.Button(card, text="🔍 วิเคราะห์ตอนนี้",
                               font=('Segoe UI', 11, 'bold'),
                               bg='#00d4ff', fg='#000000',
                               relief='flat', cursor='hand2',
                               command=self.analyze_manual)
        analyze_btn.pack(fill='x', padx=20, pady=(0, 15), ipady=8)
        
        # Status
        self.bot_status_label = tk.Label(card, text="⏸️ บอทไม่ทำงาน",
                                         font=('Segoe UI', 10),
                                         bg='#151b3d', fg='#ffa502')
        self.bot_status_label.pack(pady=(0, 10))

        # Bind scroll wheel recursively to all children after build
        card.after(100, lambda: _bind_ctrl_scroll(card))

    def create_console_panel(self, parent):
        """Create console output panel"""
        card = tk.Frame(parent, bg='#151b3d', relief='flat')
        card.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Console display
        self.console_text = scrolledtext.ScrolledText(card, height=30,
                                                      font=('Consolas', 9),
                                                      bg='#0a0e27', fg='#00ff88',
                                                      relief='flat', wrap='word')
        self.console_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure tags
        self.console_text.tag_config('error', foreground='#ff4757')
        self.console_text.tag_config('warning', foreground='#ffa502')
        self.console_text.tag_config('info', foreground='#00d4ff')
        self.console_text.tag_config('success', foreground='#00ff88')
        
        # Redirect print to console (จำกัด 500 บรรทัด, auto-clear)
        sys.stdout = ConsoleRedirector(self.console_text, 'success', max_lines=500)
        sys.stderr = ConsoleRedirector(self.console_text, 'error', max_lines=500)
        
    def create_status_bar(self):
        """Create status bar"""
        status_bar = tk.Frame(self.root, bg='#151b3d', height=30)
        status_bar.pack(side='bottom', fill='x')
        status_bar.pack_propagate(False)
        
        self.status_text = tk.Label(status_bar, text="พร้อมเชื่อมต่อ...",
                                    font=('Segoe UI', 9), bg='#151b3d',
                                    fg='#ffffff', anchor='w')
        self.status_text.pack(side='left', padx=15)
        
        self.time_label = tk.Label(status_bar, text="",
                                   font=('Segoe UI', 9), bg='#151b3d',
                                   fg='#a0a0a0', anchor='e')
        self.time_label.pack(side='right', padx=15)
        self.update_time()
        
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
        
    def log(self, message, level='info'):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert('end', log_message, level)
        self.log_text.see('end')

        # ส่งไปยัง Dashboard activity feed ด้วย
        if hasattr(self, '_dash_log_buffer') and hasattr(self, 'dash_activity_labels'):
            color_map = {'success': '#00ff88', 'error': '#ff4757',
                         'warning': '#ffa502', 'info': '#00d4ff'}
            self.dash_add_activity(message, color_map.get(level, '#a0a0a0'))
        
    def connect_mt5(self):
        """เชื่อมต่อ MT5"""
        self.log("กำลังพยายามเชื่อมต่อ MT5...", 'info')
        
        def connect():
            success = self.trader.connect()
            self.root.after(0, lambda: self.on_connect_complete(success))
        
        threading.Thread(target=connect, daemon=True).start()
        
    def on_connect_complete(self, success):
        """จัดการเมื่อเชื่อมต่อสำเร็จ"""
        if success:
            self.is_connected = True
            self.connection_label.config(text="● เชื่อมต่อแล้ว", fg='#00ff88')
            self.connect_btn.config(text="🔌 เชื่อมต่อแล้ว", state='disabled')
            self.status_text.config(text="เชื่อมต่อ MT5 สำเร็จแล้ว")
            self.log("เชื่อมต่อ MT5 สำเร็จ!", 'success')
            
            # Initialize bot
            account_info = self.trader.get_account_info()
            if account_info:
                balance = account_info['balance']
                self.log(f"ยอดเงินในบัญชี: ${balance:,.2f}", 'info')
                
                symbols = [s.strip() for s in self.bot_symbol_var.get().split(',')]
                risk_percent = float(self.bot_risk_var.get())
                fixed_volume = float(self.fixed_volume_var.get())
                
                self.bot = TradingBot(symbols=symbols, risk_percent=risk_percent, fixed_volume=fixed_volume)
                self.bot.initialize(balance)
                self.bot.min_quality_score = float(self.min_quality_var.get())
                
                # 🛡️ อัพเดทการตั้งค่าระบบป้องกันจาก UI
                self.bot.risk_manager.max_daily_trades = int(self.max_daily_trades_var.get())
                self.bot.risk_manager.daily_loss_limit_percent = float(self.daily_loss_limit_var.get())
                self.bot.risk_manager.max_drawdown_percent = float(self.max_drawdown_var.get())
                self.bot.risk_manager.max_consecutive_losses = int(self.max_consecutive_losses_var.get())
                self.bot.risk_manager.max_loss_per_trade = float(self.max_loss_per_trade_var.get())
                
                # 🛡️ แสดงสถานะระบบป้องกัน
                self.log("=" * 50, 'info')
                self.log("🛡️  ระบบป้องกันความเสี่ยงขั้นสูง 2026:", 'info')
                self.log(f"  ⛔ จำกัดขาดทุนต่อวัน: {self.bot.risk_manager.daily_loss_limit_percent}%", 'warning')
                self.log(f"  ⛔ จำกัด Drawdown สูงสุด: {self.bot.risk_manager.max_drawdown_percent}%", 'warning')
                
                daily_trades_text = f"{self.bot.risk_manager.max_daily_trades} ครั้ง" if self.bot.risk_manager.max_daily_trades > 0 else "ไม่จำกัด"
                self.log(f"  ⛔ จำกัดเทรดต่อวัน: {daily_trades_text}", 'warning')
                
                self.log(f"  ⛔ ป้องกันเสียติดกัน: {self.bot.risk_manager.max_consecutive_losses} ครั้ง", 'warning')
                
                max_loss_text = f"${self.bot.risk_manager.max_loss_per_trade:.2f}" if self.bot.risk_manager.max_loss_per_trade > 0 else "ไม่จำกัด"
                self.log(f"  ⛔ จำกัดขาดทุนต่อออเดอร์: {max_loss_text}", 'warning')
                self.log("=" * 50, 'info')
                
                self.log(f"ตั้งค่าบอทแล้ว - คู่เงิน: {', '.join(symbols)}", 'info')
        else:
            self.is_connected = False
            self.connection_label.config(text="● ล้มเหลว", fg='#ff4757')
            self.log("เชื่อมต่อ MT5 ล้มเหลว", 'error')
            messagebox.showerror("เกิดข้อผิดพลาด", "ไม่สามารถเชื่อมต่อ MT5 ได้")
    
    def analyze_manual(self):
        """วิเคราะห์ด้วยตนเอง"""
        if not self.is_connected or not self.bot:
            messagebox.showwarning("ไม่ได้เชื่อมต่อ", "กรุณาเชื่อมต่อ MT5 ก่อน!")
            return
        
        self.log("กำลังวิเคราะห์ด้วยตนเอง...", 'info')
        
        def analyze():
            symbol = self.bot_symbol_var.get().split(',')[0].strip()
            signal_report = self.bot.generate_signal(symbol)
            self.root.after(0, lambda: self.display_analysis(signal_report))
        
        threading.Thread(target=analyze, daemon=True).start()
        
    def display_analysis(self, signal_report):
        """Display analysis results"""
        self.analysis_text.delete('1.0', 'end')
        
        if 'error' in signal_report:
            self.analysis_text.insert('end', f"ERROR: {signal_report['error']}\n", 'danger')
            return
        
        summary = self.bot.get_signal_summary(signal_report)
        self.analysis_text.insert('end', summary)
        
        # Log signal info
        signal = signal_report['signal']
        direction = signal['direction']
        strength = f"{signal['strength']:.1f}%"
        
        self.log(f"วิเคราะห์เสร็จ - สัญญาณ {direction} ({strength})", 
                'success' if direction != 'NONE' else 'info')
    
    def start_auto_trading(self):
        """เริ่มเทรดอัตโนมัติ"""
        if not self.is_connected or not self.bot:
            messagebox.showwarning("ไม่ได้เชื่อมต่อ", "กรุณาเชื่อมต่อ MT5 ก่อน!")
            return
        
        # คำเตือน
        warning_msg = "⚠️ คำเตือน! ⚠️\n\n"
        warning_msg += "🚀 โหมดเร็ว (แบบเดียว)\n\n"
        warning_msg += "บอทจะ:\n"
        warning_msg += "• เปิดออเดอร์ทันทีที่เจอสัญญาณ\n"
        warning_msg += "• เทรดบ่อยมาก\n"
        warning_msg += "• ปิดออเดอร์เมื่อได้กำไร\n\n"
        warning_msg += "🛡️ Circuit Breaker จะหยุดบอทเมื่อ:\n"
        warning_msg += "• ขาดทุน 5% ต่อวัน\n"
        warning_msg += "• Drawdown เกิน 15%\n"
        
        # Get max daily trades value
        max_daily_trades = int(self.max_daily_trades_var.get())
        if max_daily_trades > 0:
            warning_msg += f"• เทรดเกิน {max_daily_trades} ครั้ง/วัน\n"
        else:
            warning_msg += "• ไม่จำกัดจำนวนครั้ง/วัน\n"
        
        warning_msg += "• ตลาดผันผวนผิดปกติ\n\n"
        warning_msg += "⚠️ แนะนำให้ใช้ Demo Account!\n\n"
        warning_msg += "คุณแน่ใจหรือไม่ที่จะดำเนินการต่อ?"
        
        confirm = messagebox.askyesno("เริ่มเทรดอัตโนมัติ", warning_msg)
        
        if not confirm:
            return
        
        # ตั้งค่าบอท - ใช้โหมดเร็วเสมอ
        self.bot.aggressive_mode = True
        self.bot.min_profit_pips = 10
        
        # ⚡ อัพเดทค่าจาก UI (กรณีผู้ใช้เปลี่ยนหลัง connect)
        symbols = [s.strip() for s in self.bot_symbol_var.get().split(',')]
        fixed_vol = float(self.fixed_volume_var.get())
        max_daily_trades = int(self.max_daily_trades_var.get())
        max_loss_per_trade = float(self.max_loss_per_trade_var.get())
        
        self.bot.symbols = symbols  # อัพเดทคู่เงิน
        self.bot.fixed_volume = fixed_vol  # อัพเดท Volume
        self.bot.risk_manager.max_daily_trades = max_daily_trades  # อัพเดทจำกัดเทรด/วัน
        self.bot.risk_manager.max_loss_per_trade = max_loss_per_trade  # อัพเดทจำกัดขาดทุนต่อออเดอร์
        
        self.log(f"📊 Symbol: {', '.join(symbols)}", 'info')
        self.log(f"📊 Fixed Volume: {fixed_vol} lots (ทุกออเดอร์)", 'info')
        self.log("🚀 เปิดโหมดเร็ว - ปิดออเดอร์ทันทีที่มีกำไร", 'warning')
        self.log("🛡️  Circuit Breakers ทำงานอยู่ - บอทจะหยุดถ้าขาดทุนเกิน!", 'warning')
        
        self.auto_trading_active = True
        self.start_bot_btn.config(state='disabled')
        self.stop_bot_btn.config(state='normal')
        self.bot_status_label.config(text="🚀 บอททำงาน (โหมดเร็ว)", fg='#ffa502')
        
        interval = int(self.scan_interval_var.get())
        
        self.log("🤖 เริ่มบอทเทรดอัตโนมัติแล้ว!", 'success')
        self.log(f"สแกนตลาดทุก {interval} วินาที", 'info')
        
        # Start bot in separate thread
        def run_bot():
            self.bot.start_auto_trading(self.trader, interval)
        
        self.bot_thread = threading.Thread(target=run_bot, daemon=True)
        self.bot_thread.start()
        
    def stop_auto_trading(self):
        """หยุดเทรดอัตโนมัติ"""
        self.auto_trading_active = False
        
        if self.bot:
            self.bot.stop_auto_trading()
        
        self.start_bot_btn.config(state='normal')
        self.stop_bot_btn.config(state='disabled')
        self.bot_status_label.config(text="⏹️ หยุดบอทแล้ว", fg='#ffa502')
        
        self.log("🛑 หยุดบอทเทรดอัตโนมัติแล้ว", 'warning')

def main():
    root = tk.Tk()
    app = AITradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
