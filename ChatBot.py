"""
ChatBot.py
Discord bot for daily account status reporting
"""
import discord
from discord.ext import commands, tasks
import asyncio
import os
from datetime import datetime, time
import pytz
from TradingBot import TradingBot

class TradingStatusBot(commands.Bot):
    """Discord bot for automated trading status reports"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(command_prefix='!', intents=intents)
        
        # Trading bot instance
        self.trading_bot = None
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))  # Set your channel ID
        
        # Initialize trading bot
        if self.api_key and self.secret_key:
            self.trading_bot = TradingBot(None, api_key=self.api_key, 
                                        secret_key=self.secret_key, paper=True)
    
    async def on_ready(self):
        """Called when bot is ready"""
        print(f'[AI] {self.user} has connected to Discord!')
        print(f'[INFO] Trading Bot Status: {"Connected" if self.trading_bot else "Disconnected"}')
        
        # Start daily reporting task
        if not self.daily_report.is_running():
            self.daily_report.start()
            print("⏰ Daily reporting task started (7:00 AM)")
    
    @tasks.loop(time=time(7, 0, tzinfo=pytz.timezone('US/Eastern')))  # 7:00 AM EST
    async def daily_report(self):
        """Send daily account status report"""
        
        if not self.trading_bot or self.channel_id == 0:
            print("[WARNING] Cannot send daily report: Trading bot or channel not configured")
            return
        
        try:
            channel = self.get_channel(self.channel_id)
            if not channel:
                print(f"[ERROR] Could not find channel with ID: {self.channel_id}")
                return
            
            # Generate comprehensive report
            report = await self.generate_daily_report()
            
            # Send report as embed
            embed = discord.Embed(
                title="[INFO] Daily Trading Account Status",
                description=f"Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S EST')}",
                color=0x00ff00  # Green
            )
            
            # Add fields from report
            for section, content in report.items():
                embed.add_field(name=section, value=content, inline=False)
            
            embed.set_footer(text="Automated Trading Bot • Daily Report")
            
            await channel.send(embed=embed)
            print(f"[SUCCESS] Daily report sent to channel {channel.name}")
            
        except Exception as e:
            print(f"[ERROR] Error sending daily report: {e}")
    
    async def generate_daily_report(self) -> dict:
        """Generate comprehensive daily trading report"""
        
        try:
            # Get account information
            account_info = self.trading_bot.get_account_info()
            positions = self.trading_bot.get_positions()
            
            report = {
                "[INFO] Account Summary": f"""
                **Portfolio Value:** ${float(account_info.get('portfolio_value', 0)):,.2f}
                **Cash Available:** ${float(account_info.get('cash', 0)):,.2f}
                **Day P&L:** ${float(account_info.get('unrealized_pl', 0)):,.2f}
                **Total P&L:** ${float(account_info.get('unrealized_pl', 0)):,.2f}
                """,
                
                "[INFO] Current Positions": self._format_positions(positions),
                
                "[INFO] Market Status": f"""
                **Market Hours:** {"Open" if self._is_market_open() else "Closed"}
                **Account Status:** {account_info.get('status', 'Unknown')}
                **Buying Power:** ${float(account_info.get('buying_power', 0)):,.2f}
                """,
                
                "[INFO] Trading Activity": "No recent activity",  # Implement if needed
                
                "[WARNING] Alerts": self._check_alerts(account_info, positions)
            }
            
            return report
            
        except Exception as e:
            return {"[ERROR] Error": f"Failed to generate report: {str(e)}"}
    
    def _format_positions(self, positions) -> str:
        """Format positions for display"""
        if not positions:
            return "No current positions"
        
        formatted = []
        for position in positions[:10]:  # Limit to 10 positions
            symbol = position.get('symbol', 'Unknown')
            qty = float(position.get('qty', 0))
            market_value = float(position.get('market_value', 0))
            unrealized_pl = float(position.get('unrealized_pl', 0))
            
            pl_status = "[UP]" if unrealized_pl >= 0 else "[DOWN]"
            formatted.append(f"{pl_status} **{symbol}**: {qty:,.0f} shares (${market_value:,.2f})")
        
        return "\n".join(formatted) if formatted else "No positions"
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            # Simple market hours check (9:30 AM - 4:00 PM EST, weekdays)
            now = datetime.now(pytz.timezone('US/Eastern'))
            if now.weekday() >= 5:  # Weekend
                return False
            
            market_open = time(9, 30)
            market_close = time(16, 0)
            current_time = now.time()
            
            return market_open <= current_time <= market_close
        except:
            return False
    
    def _check_alerts(self, account_info, positions) -> str:
        """Check for important alerts"""
        alerts = []
        
        try:
            # Check for low cash
            cash = float(account_info.get('cash', 0))
            if cash < 1000:
                alerts.append("🔴 Low cash balance")
            
            # Check for large losses
            day_pl = float(account_info.get('unrealized_pl', 0))
            if day_pl < -500:
                alerts.append("🔴 Significant daily loss")
            
            # Check for account restrictions
            if account_info.get('status') != 'ACTIVE':
                alerts.append("🔴 Account status issue")
            
        except Exception as e:
            alerts.append(f"[WARNING] Error checking alerts: {str(e)}")
        
        return "\n".join(alerts) if alerts else "[SUCCESS] No alerts"
    
    # Bot commands
    @commands.command(name='status')
    async def status_command(self, ctx):
        """Manual status check command"""
        
        if not self.trading_bot:
            await ctx.send("[ERROR] Trading bot not connected")
            return
        
        try:
            report = await self.generate_daily_report()
            
            embed = discord.Embed(
                title="[INFO] Current Account Status",
                color=0x0099ff  # Blue
            )
            
            for section, content in report.items():
                embed.add_field(name=section, value=content, inline=False)
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"[ERROR] Error generating status: {str(e)}")
    
    @commands.command(name='positions')
    async def positions_command(self, ctx):
        """Get current positions"""
        
        if not self.trading_bot:
            await ctx.send("[ERROR] Trading bot not connected")
            return
        
        try:
            positions = self.trading_bot.get_positions()
            formatted_positions = self._format_positions(positions)
            
            embed = discord.Embed(
                title="[INFO] Current Positions",
                description=formatted_positions,
                color=0x00ff00
            )
            
            await ctx.send(embed=embed)
            
        except Exception as e:
            await ctx.send(f"[ERROR] Error getting positions: {str(e)}")

def run_discord_bot():
    """Run the Discord bot"""
    
    # Get Discord bot token from environment
    token = os.getenv('DISCORD_BOT_TOKEN')
    
    if not token:
        print("[ERROR] DISCORD_BOT_TOKEN not found in environment variables")
        print("[TIP] Set the following environment variables:")
        print("   - DISCORD_BOT_TOKEN")
        print("   - DISCORD_CHANNEL_ID")
        print("   - ALPACA_API_KEY")
        print("   - ALPACA_SECRET_KEY")
        return
    
    bot = TradingStatusBot()
    
    try:
        bot.run(token)
    except Exception as e:
        print(f"[ERROR] Error running Discord bot: {e}")

if __name__ == "__main__":
    run_discord_bot()
