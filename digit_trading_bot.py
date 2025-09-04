import logging
import csv
from datetime import datetime, timedelta
from collections import deque
import json
import websockets
import asyncio
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DigitTradingConfig:
    """Configuration for digit trading bot"""
    # Trade Settings
    stake: float = 1.0                    # Configurable stake amount
    target_digit: int = 9                 # Digit to watch for (configurable)
    trade_digit: int = 8                  # Digit to trade when target appears
    contract_duration: int = 5            # Number of ticks for contract
    symbol: str = 'R_50'                 # Trading symbol
    
    # Risk Management
    max_trades_per_hour: int = 130         # Maximum trades per hour
    max_open_contracts: int = 2           # Maximum simultaneous contracts
    daily_profit_target: float = 1.0   # Stop when target is reached
    daily_loss_limit: float = 1.0       # Stop loss limit
    
    # Trading Logic
    min_wait_seconds: int = 10            # Minimum wait between trades
    consecutive_target_required: int = 1   # How many consecutive target digits needed
    
    # Account Settings
    use_real_account: bool = True        # Set to True for real account
    
    # Connection Settings
    websocket_timeout: int = 60
    
    @property
    def app_id(self) -> int:
        """Get appropriate app_id based on account type"""
        if self.use_real_account:
            return 99240  
        else:
            return 85574  # Demo account app_id
    
    @property
    def api_token(self) -> str:
        """Get appropriate API token based on account type"""
        if self.use_real_account:
            return ""
        else:
            return "lUih4ezjsQwFivN"
    
    @property
    def connection_url(self) -> str:
        return f'wss://ws.derivws.com/websockets/v3?app_id={self.app_id}'

class DigitAnalyzer:
    """Analyzes digit patterns for trading signals"""
    
    def __init__(self, config: DigitTradingConfig):
        self.config = config
        self.recent_digits = deque(maxlen=20)  # Store recent second last digits
        self.target_digit_count = 0
        
    def extract_last_digit(self, price: float) -> int:
        """Extract the second last digit from price"""
        price_str = f"{price:.5f}"  # Ensure enough decimal places
        # Get the second last digit from the price string (excluding decimal point)
        digits_only = price_str.replace('.', '')
        # Return second last digit (index -2) if available, otherwise last digit
        return int(digits_only[-2]) if len(digits_only) >= 2 else int(digits_only[-1]) if digits_only else 0
    
    def add_price(self, price: float) -> None:
        """Add new price and extract digit"""
        last_digit = self.extract_last_digit(price)
        self.recent_digits.append(last_digit)
        
        # Count consecutive target digits
        if last_digit == self.config.target_digit:
            self.target_digit_count += 1
        else:
            self.target_digit_count = 0
    
    def should_trade(self) -> Tuple[bool, str]:
        """Check if we should place a trade based on digit pattern"""
        if len(self.recent_digits) < self.config.consecutive_target_required:
            return False, "Not enough data"
        
        # Check if we have the required consecutive target digits
        if self.target_digit_count >= self.config.consecutive_target_required:
            # We've seen the target digit, now trade UNDER the trade_digit
            signal_type = f"DIGITUNDER_{self.config.trade_digit}"
            reason = f"Seen {self.target_digit_count} consecutive {self.config.target_digit}(s), trading UNDER {self.config.trade_digit}"
            return True, reason
        
        return False, f"Waiting for {self.config.target_digit} digit pattern"
    
    def get_recent_digits_summary(self) -> str:
        """Get summary of recent digits for logging"""
        if not self.recent_digits:
            return "No digits yet"
        return f"Last 5: {list(self.recent_digits)[-5:]}, Target {self.config.target_digit} count: {self.target_digit_count}"

class DigitProfitTracker:
    """Track profit/loss for digit trading"""
    
    def __init__(self, config: DigitTradingConfig):
        self.config = config
        self.daily_pnl = 0
        self.trade_count = 0
        self.win_count = 0
        self.last_reset = datetime.now().date()
        self.trade_history = deque(maxlen=50)
        
    def reset_daily(self):
        """Reset daily counters"""
        current_date = datetime.now().date()
        if current_date != self.last_reset:
            logging.info(f"📊 Daily Reset - PnL: ${self.daily_pnl:.2f}, Trades: {self.trade_count}, Win Rate: {self.get_win_rate():.1%}")
            self.daily_pnl = 0
            self.trade_count = 0
            self.win_count = 0
            self.last_reset = current_date
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on profit/loss limits"""
        self.reset_daily()
        
        if self.daily_pnl >= self.config.daily_profit_target:
            return False, f"✅ Daily profit target (${self.config.daily_profit_target}) reached!"
        
        if self.daily_pnl <= -self.config.daily_loss_limit:
            return False, f"🛑 Daily loss limit (${self.config.daily_loss_limit}) reached!"
        
        return True, "Ready to trade"
    
    def record_trade(self, pnl: float):
        """Record trade result"""
        self.daily_pnl += pnl
        self.trade_count += 1
        
        if pnl > 0:
            self.win_count += 1
        
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'daily_pnl': self.daily_pnl
        })
        
        result_emoji = "🟢" if pnl > 0 else "🔴"
        logging.info(f"{result_emoji} Trade Result: ${pnl:.2f} | Daily PnL: ${self.daily_pnl:.2f} | Win Rate: {self.get_win_rate():.1%}")
    
    def get_win_rate(self) -> float:
        """Calculate current win rate"""
        return self.win_count / max(self.trade_count, 1)

class DigitTradeManager:
    """Manage digit trading operations"""
    
    def __init__(self, config: DigitTradingConfig):
        self.config = config
        self.last_trade_time = datetime.now() - timedelta(hours=1)
        self.open_contracts = set()
        self.profit_tracker = DigitProfitTracker(config)
        
    def can_place_trade(self) -> Tuple[bool, str]:
        """Check if we can place a new trade"""
        # Check profit/loss limits
        can_trade_profit, profit_msg = self.profit_tracker.can_trade()
        if not can_trade_profit:
            return False, profit_msg
        
        # Check open contracts limit
        if len(self.open_contracts) >= self.config.max_open_contracts:
            return False, f"Max contracts ({self.config.max_open_contracts}) open"
        
        # Check timing constraints
        time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
        min_interval = max(self.config.min_wait_seconds, 3600 / self.config.max_trades_per_hour)
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            return False, f"Wait {wait_time:.0f}s for next trade"
        
        return True, "Ready to place trade"

class DigitWebSocketManager:
    """WebSocket management for digit trading"""
    
    def __init__(self, config: DigitTradingConfig):
        self.config = config
        
    async def authenticate(self, websocket) -> None:
        """Authenticate with Deriv API"""
        auth_request = {"authorize": self.config.api_token}
        await websocket.send(json.dumps(auth_request))
        response = await websocket.recv()
        data = json.loads(response)
        
        if 'error' in data:
            error_msg = data['error']['message']
            logging.error(f"❌ Authentication error: {error_msg}")
            raise Exception(f"Authentication failed: {error_msg}")
        
        logging.info("✅ Authentication successful")
    
    async def place_digit_trade(self, websocket, contract_type: str) -> Optional[str]:
        """Place a digit trade (UNDER)"""
        buy_request = {
            "buy": 1,
            "subscribe": 1,
            "price": self.config.stake,
            "parameters": {
                "amount": self.config.stake,
                "basis": "stake",
                "contract_type": contract_type,  # e.g., "DIGITUNDER"
                "currency": "USD",
                "duration": self.config.contract_duration,
                "duration_unit": "t",
                "symbol": self.config.symbol,
                "barrier": str(self.config.trade_digit)  # The digit barrier (e.g., "8")
            }
        }
        
        try:
            await websocket.send(json.dumps(buy_request))
            response = await websocket.recv()
            data = json.loads(response)
            
            if 'error' in data:
                logging.error(f"❌ Trade error: {data['error']['message']}")
                return None
            
            contract_id = data.get('buy', {}).get('contract_id')
            if contract_id:
                buy_price = float(data.get('buy', {}).get('buy_price', self.config.stake))
                logging.info(f"🎯 Trade Placed: {contract_type} {self.config.trade_digit}, ID: {contract_id}, Cost: ${buy_price:.2f}")
            
            return contract_id
            
        except Exception as e:
            logging.error(f"❌ Error placing trade: {e}")
            return None

class DigitDataLogger:
    """Log digit trading data"""
    
    def __init__(self, filename: str = 'digit_trades.csv'):
        self.filename = filename
        self._ensure_header()
    
    def _ensure_header(self):
        """Ensure CSV has proper header"""
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'price', 'last_digit', 'signal', 'reason'])
    
    def log_trade(self, price: float, last_digit: int, signal: str, reason: str):
        """Log trade data"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, price, last_digit, signal, reason])

class DigitTradingBot:
    """Main digit trading bot"""
    
    def __init__(self, config: DigitTradingConfig):
        self.config = config
        self.analyzer = DigitAnalyzer(config)
        self.trade_manager = DigitTradeManager(config)
        self.websocket_manager = DigitWebSocketManager(config)
        self.logger = DigitDataLogger()
        
    async def run(self):
        """Main trading loop"""
        logging.info("🚀 Starting Digit Trading Bot...")
        logging.info(f"Strategy: When last digit = {self.config.target_digit}, trade UNDER {self.config.trade_digit}")
        logging.info(f"Stake: ${self.config.stake}, Target: ${self.config.daily_profit_target}")
        
        # Setup tick subscription
        ticks_request = {
            "ticks": self.config.symbol,
            "subscribe": 1
        }
        
        while True:
            try:
                async with websockets.connect(
                    self.config.connection_url,
                    ping_interval=20,
                    ping_timeout=self.config.websocket_timeout
                ) as websocket:
                    await self.websocket_manager.authenticate(websocket)
                    await websocket.send(json.dumps(ticks_request))
                    
                    logging.info("✅ Connected and monitoring digits...")
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._process_message(data, websocket)
                        except Exception as e:
                            logging.error(f"❌ Error processing message: {e}")
                            
            except (websockets.ConnectionClosed, Exception) as e:
                logging.error(f"💥 Connection error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
    
    async def _process_message(self, data: dict, websocket):
        """Process incoming WebSocket messages"""
        if 'error' in data:
            logging.error(f"❌ API Error: {data['error']['message']}")
            return
        
        # Handle contract updates
        if data.get('msg_type') == 'proposal_open_contract':
            await self._handle_contract_update(data)
            return
        
        # Handle tick data
        if data.get('msg_type') != 'tick':
            return
        
        tick_data = data.get('tick', {})
        price = float(tick_data.get('quote', 0))
        
        if price <= 0:
            return
        
        # Analyze the digit
        self.analyzer.add_price(price)
        last_digit = self.analyzer.extract_last_digit(price)
        
        # Log progress every 20 ticks
        if len(self.analyzer.recent_digits) % 20 == 0:
            daily_pnl = self.trade_manager.profit_tracker.daily_pnl
            logging.info(f"📊 Price: {price:.5f}, Last Digit: {last_digit}, {self.analyzer.get_recent_digits_summary()}, Daily PnL: ${daily_pnl:.2f}")
        
        # Check if we should trade
        should_trade, reason = self.analyzer.should_trade()
        
        if should_trade:
            can_trade, trade_msg = self.trade_manager.can_place_trade()
            
            if can_trade:
                logging.info(f"🎯 Trading Signal: {reason}")
                
                # Place UNDER trade
                contract_type = f"DIGITUNDER"
                contract_id = await self.websocket_manager.place_digit_trade(websocket, contract_type)
                
                if contract_id:
                    self.trade_manager.open_contracts.add(contract_id)
                    self.trade_manager.last_trade_time = datetime.now()
                    self.logger.log_trade(price, last_digit, contract_type, reason)
                    
                    # Subscribe to contract updates
                    await self._subscribe_to_contract(websocket, contract_id)
                    
                    # Reset the target digit count after placing trade
                    self.analyzer.target_digit_count = 0
                    
            else:
                if "profit target" in trade_msg.lower() or "loss limit" in trade_msg.lower():
                    logging.info(f"🛑 {trade_msg}")
    
    async def _subscribe_to_contract(self, websocket, contract_id: str):
        """Subscribe to contract updates"""
        try:
            request = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1
            }
            await websocket.send(json.dumps(request))
        except Exception as e:
            logging.error(f"❌ Error subscribing to contract {contract_id}: {e}")
    
    async def _handle_contract_update(self, data: dict):
        """Handle contract completion and profit calculation"""
        try:
            contract = data.get('proposal_open_contract', {})
            contract_id = contract.get('contract_id')
            status = contract.get('status')
            
            if not contract_id or contract_id not in self.trade_manager.open_contracts:
                return
            
            if status in ['sold', 'won', 'lost']:
                # Calculate profit/loss
                buy_price = float(contract.get('buy_price', self.config.stake))
                sell_price = float(contract.get('sell_price', 0))
                pnl = sell_price - buy_price
                
                # Record the result
                self.trade_manager.profit_tracker.record_trade(pnl)
                self.trade_manager.open_contracts.discard(contract_id)
                
                logging.info(f"📋 Contract {contract_id}: {status.upper()}, PnL: ${pnl:.2f}")
                
        except Exception as e:
            logging.error(f"❌ Error handling contract update: {e}")

async def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('digit_trading.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create configuration (easily customizable)
    config = DigitTradingConfig(
        stake=5.0,          # Adjust stake amount
        target_digit=9,         # Digit to watch for
        trade_digit=8,          # Trade UNDER this digit
        contract_duration=1,    # Contract duration in ticks
        daily_profit_target=2.0,
        daily_loss_limit=1.0,
        use_real_account=True  # Set to True for REAL ACCOUNT trading
    )
    
    # Display configuration
    account_type = "🔴 REAL ACCOUNT" if config.use_real_account else "🟡 DEMO ACCOUNT"
    print("🎲 DIGIT TRADING BOT")
    print("=" * 50)
    print(f"Account Type: {account_type}")
    print(f"Strategy: When last digit = {config.target_digit}, trade UNDER {config.trade_digit}")
    print(f"Symbol: {config.symbol}")
    print(f"Stake: ${config.stake}")
    print(f"Contract Duration: {config.contract_duration} ticks")
    print(f"Daily Target: ${config.daily_profit_target}")
    print(f"Daily Limit: ${config.daily_loss_limit}")
    print("=" * 50)
    
    # Safety warning for real account
    if config.use_real_account:
        print("⚠️  WARNING: YOU ARE USING A REAL ACCOUNT!")
        print("⚠️  REAL MONEY WILL BE AT RISK!")
        print("⚠️  Make sure you have set appropriate stake amounts and limits!")
        print("=" * 50)
    
    print()
    
    # Create and run bot
    bot = DigitTradingBot(config)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logging.info("🛑 Bot stopped by user")
    except Exception as e:
        logging.error(f"💥 Bot crashed: {e}")

if __name__ == "__main__":
    asyncio.run(main())