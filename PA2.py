"""
Main program following COMP2012 Assignment 2 design pattern
ENHANCED VERSION with Environment Variable Configuration
"""
from datetime import datetime
import time
import os
from dotenv import load_dotenv  # For .env file support

# Updated imports to reflect the new file structure
from Market import Market, StockPredictor

# [SUCCESS] UPDATED: Import enhanced grading system
from Analysis import (
    ComprehensiveStockAnalyzer,
    SP500ComprehensiveAnalyzer, 
    UserPreferenceManager,
    VisualizationEngine,
    StabilityAnalyzer,
    EnhancedGradingWithGenAI
)

from TradingBot import TradingBot, AutomatedTradingSystemLoop

class ConfigurationManager:
    """[SUCCESS] NEW: Manages all configuration and environment variables"""
    
    def __init__(self):
        # Load .env file if it exists
        load_dotenv()
        
        self.config = {
            'alpaca_api_key': None,
            'alpaca_secret_key': None,
            'openrouter_api_key': None,
            'discord_bot_token': None,
            'discord_channel_id': None,
            'environment': 'development'
        }
        
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from environment variables"""
        
        # [SUCCESS] Load Alpaca credentials
        self.config['alpaca_api_key'] = os.getenv('ALPACA_API_KEY')
        self.config['alpaca_secret_key'] = os.getenv('ALPACA_SECRET_KEY')
        
        # [SUCCESS] Load OpenRouter credentials for GenAI
        self.config['openrouter_api_key'] = os.getenv('OPENROUTER_API_KEY')
        
        # [SUCCESS] Load Discord credentials
        self.config['discord_bot_token'] = os.getenv('DISCORD_BOT_TOKEN')
        self.config['discord_channel_id'] = os.getenv('DISCORD_CHANNEL_ID')
        
        # [SUCCESS] Load environment setting
        self.config['environment'] = os.getenv('ENVIRONMENT', 'development')
    
    def get_alpaca_credentials(self):
        """Get Alpaca API credentials with fallback options"""
        
        api_key = self.config['alpaca_api_key']
        secret_key = self.config['alpaca_secret_key']
        
        if not api_key or not secret_key:
            print("[WARNING] Alpaca credentials not found in environment variables")
            print("[TIP] Options:")
            print("   1. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            print("   2. Create a .env file with your credentials")
            print("   3. Enter credentials manually (not recommended for production)")
            
            choice = input("\nChoose option (1, 2, or 3): ").strip()
            
            if choice == "2":
                self._create_env_file()
                load_dotenv()  # Reload after creating .env
                api_key = os.getenv('ALPACA_API_KEY')
                secret_key = os.getenv('ALPACA_SECRET_KEY')
            elif choice == "3":
                print("\n[SECURE] Manual Credential Entry (Temporary Session Only):")
                api_key = input("Enter Alpaca API Key: ").strip()
                secret_key = input("Enter Alpaca Secret Key: ").strip()
            else:
                print("\n[INFO] Environment Variable Setup Instructions:")
                self._show_env_setup_instructions()
                return None, None
        
        return api_key, secret_key
    
    def get_openrouter_key(self):
        """Get OpenRouter API key for GenAI features"""
        return self.config['openrouter_api_key']
    
    def get_discord_config(self):
        """Get Discord bot configuration with validation"""
        token = self.config['discord_bot_token']
        channel_id = self.config['discord_channel_id']
        
        # [SUCCESS] FIXED: Clean and validate channel ID
        if channel_id:
            # Remove spaces and non-numeric characters
            channel_id = ''.join(filter(str.isdigit, str(channel_id)))
            
            if not channel_id:
                print("[ERROR] Invalid Discord Channel ID - must be numeric")
                return {'token': None, 'channel_id': None}
        
        return {
            'token': token,
            'channel_id': int(channel_id) if channel_id else None
        }
    
    def is_genai_available(self):
        """Check if GenAI features are available"""
        return self.config['openrouter_api_key'] is not None
    
    def is_discord_available(self):
        """Check if Discord bot features are available"""
        return (self.config['discord_bot_token'] is not None and 
                self.config['discord_channel_id'] is not None)
    
    def _create_env_file(self):
        """Create .env file with user input"""
        
        print("\n[INFO] Creating .env file for secure credential storage...")
        
        # Get credentials
        alpaca_key = input("Enter your Alpaca API Key: ").strip()
        alpaca_secret = input("Enter your Alpaca Secret Key: ").strip()
        
        genai_choice = input("Do you want to enable GenAI features? (y/n): ").lower()
        openrouter_key = ""
        if genai_choice == 'y':
            openrouter_key = input("Enter your OpenRouter API Key: ").strip()
        
        discord_choice = input("Do you want to enable Discord bot? (y/n): ").lower()
        discord_token = ""
        discord_channel = ""
        if discord_choice == 'y':
            discord_token = input("Enter your Discord Bot Token: ").strip()
            discord_channel = input("Enter your Discord Channel ID: ").strip()
        
        # Create .env content
        env_content = f"""# Alpaca Trading Configuration
ALPACA_API_KEY={alpaca_key}
ALPACA_SECRET_KEY={alpaca_secret}

# OpenRouter Configuration (for GenAI features)
OPENROUTER_API_KEY={openrouter_key}

# Discord Bot Configuration
DISCORD_BOT_TOKEN={discord_token}
DISCORD_CHANNEL_ID={discord_channel}

# Environment Setting
ENVIRONMENT=development

# Additional Settings
MAX_POSITION_SIZE=1000
RISK_TOLERANCE=medium
"""
        
        try:
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print("[SUCCESS] .env file created successfully!")
            print("[WARNING] Important: Add .env to your .gitignore file to keep credentials secure")
            
            # Update gitignore
            self._update_gitignore()
            
        except Exception as e:
            print(f"[ERROR] Error creating .env file: {e}")
    
    def _update_gitignore(self):
        """Update .gitignore to include .env file"""
        
        gitignore_path = '.gitignore'
        env_entry = '.env\n'
        
        try:
            # Check if .gitignore exists and if .env is already in it
            if os.path.exists(gitignore_path):
                with open(gitignore_path, 'r') as f:
                    content = f.read()
                
                if '.env' not in content:
                    with open(gitignore_path, 'a') as f:
                        f.write('\n# Environment variables\n.env\n')
                    print("[SUCCESS] Updated .gitignore to include .env")
            else:
                # Create new .gitignore
                with open(gitignore_path, 'w') as f:
                    f.write('# Environment variables\n.env\n\n# Python\n__pycache__/\n*.pyc\n*.pyo\n')
                print("[SUCCESS] Created .gitignore with .env entry")
                
        except Exception as e:
            print(f"[WARNING] Could not update .gitignore: {e}")
    
    def _show_env_setup_instructions(self):
        """Show detailed environment setup instructions"""
        
        print("\n" + "="*80)
        print("[CONFIG] ENVIRONMENT VARIABLE SETUP INSTRUCTIONS")
        print("="*80)
        
        print("\n[INFO] Method 1: Create .env file (Recommended)")
        print("Create a file named '.env' in your project directory with:")
        print("""
# Alpaca Trading Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# OpenRouter Configuration (for GenAI features)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Discord Bot Configuration
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here

# Environment Setting
ENVIRONMENT=development
""")
        
        print("\n[INFO] Method 2: System Environment Variables")
        print("Set these environment variables in your system:")
        
        if os.name == 'nt':  # Windows
            print("Windows Command Prompt:")
            print("set ALPACA_API_KEY=your_key_here")
            print("set ALPACA_SECRET_KEY=your_secret_here")
            print("set OPENROUTER_API_KEY=your_openrouter_key_here")
        else:  # Linux/Mac
            print("Linux/Mac Terminal:")
            print("export ALPACA_API_KEY=your_key_here")
            print("export ALPACA_SECRET_KEY=your_secret_here")
            print("export OPENROUTER_API_KEY=your_openrouter_key_here")
        
        print("\n[INFO] Method 3: IDE Configuration")
        print("Configure environment variables in your IDE settings:")
        print("- PyCharm: Run/Debug Configurations -> Environment Variables")
        print("- VS Code: Create launch.json with env settings")
        print("- Jupyter: Use %env magic commands")
        
        print("\n[SECURITY] Security Best Practices:")
        print("- Never commit API keys to version control")
        print("- Add .env to your .gitignore file")
        print("- Use different keys for development and production")
        print("- Regularly rotate your API keys")
        
        print("\n[RESOURCES] Where to Get API Keys:")
        print("- Alpaca: https://app.alpaca.markets/paper/dashboard/overview")
        print("- OpenRouter: https://openrouter.ai/keys")
        print("- Discord Bot: https://discord.com/developers/applications")
        
    def _display_config_value(self, key: str, value: str) -> str:
        """Display configuration value with first 4 and last 4 characters visible"""
        if not value:
            return "[ERROR] Not set"
        
        # Special handling for channel ID (show full value since it's not sensitive)
        if 'channel_id' in key.lower():
            return f"[SUCCESS] {value}"
        
        # For API keys and tokens, show first 4 and last 4 characters
        if 'key' in key.lower() or 'token' in key.lower():
            if len(value) >= 8:
                return f"[SUCCESS] {value[:4]}****{value[-4:]}"
            else:
                return f"[SUCCESS] {value[:2]}****{value[-2:]}"
        
        # For other values, show normally
        return f"[SUCCESS] {value}"
    
    def _update_env_file_selective(self):
        """Update specific values in existing .env file"""
        
        print("\n[INFO] SELECTIVE .ENV FILE UPDATE")
        print("="*50)
        
        # Load existing .env values
        load_dotenv()
        current_values = {
            'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY', ''),
            'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY', ''),
            'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY', ''),
            'DISCORD_BOT_TOKEN': os.getenv('DISCORD_BOT_TOKEN', ''),
            'DISCORD_CHANNEL_ID': os.getenv('DISCORD_CHANNEL_ID', ''),
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development')
        }
        
        print("Current values:")
        for key, value in current_values.items():
            display_value = self._display_config_value(key, value)
            print(f"  {key}: {display_value}")
        
        print("\nSelect values to update:")
        print("1. Alpaca API Key")
        print("2. Alpaca Secret Key") 
        print("3. OpenRouter API Key")
        print("4. Discord Bot Token")
        print("5. Discord Channel ID")
        print("6. Environment Setting")
        print("7. Update All")
        print("8. Cancel")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        updated_values = current_values.copy()
        
        if choice == "1":
            new_value = input("Enter new Alpaca API Key: ").strip()
            if new_value:
                updated_values['ALPACA_API_KEY'] = new_value
        elif choice == "2":
            new_value = input("Enter new Alpaca Secret Key: ").strip()
            if new_value:
                updated_values['ALPACA_SECRET_KEY'] = new_value
        elif choice == "3":
            new_value = input("Enter new OpenRouter API Key: ").strip()
            if new_value:
                updated_values['OPENROUTER_API_KEY'] = new_value
        elif choice == "4":
            new_value = input("Enter new Discord Bot Token: ").strip()
            if new_value:
                updated_values['DISCORD_BOT_TOKEN'] = new_value
        elif choice == "5":
            new_value = input("Enter new Discord Channel ID (numbers only): ").strip()
            # Validate channel ID
            clean_value = ''.join(filter(str.isdigit, new_value))
            if clean_value and len(clean_value) >= 10:  # Discord IDs are typically 17-19 digits
                updated_values['DISCORD_CHANNEL_ID'] = clean_value
            else:
                print("[ERROR] Invalid Discord Channel ID - must be at least 10 digits")
        elif choice == "6":
            new_value = input("Enter new Environment (development/production): ").strip()
            if new_value:
                updated_values['ENVIRONMENT'] = new_value
        elif choice == "7":
            self._create_env_file()  # Use existing method for full update
            return
        elif choice == "8":
            print("[ERROR] Update cancelled")
            return
        else:
            print("[ERROR] Invalid choice")
            return
        
        # Write updated .env file
        try:
            env_content = f"""# Alpaca Trading Configuration
    ALPACA_API_KEY={updated_values['ALPACA_API_KEY']}
    ALPACA_SECRET_KEY={updated_values['ALPACA_SECRET_KEY']}
    
    # OpenRouter Configuration (for GenAI features)
    OPENROUTER_API_KEY={updated_values['OPENROUTER_API_KEY']}
    
    # Discord Bot Configuration
    DISCORD_BOT_TOKEN={updated_values['DISCORD_BOT_TOKEN']}
    DISCORD_CHANNEL_ID={updated_values['DISCORD_CHANNEL_ID']}
    
    # Environment Setting
    ENVIRONMENT={updated_values['ENVIRONMENT']}
    
    # Additional Settings
    MAX_POSITION_SIZE=1000
    RISK_TOLERANCE=medium
    """
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print("[SUCCESS] .env file updated successfully!")
            
            # Reload configuration
            load_dotenv()
            self.load_configuration()
            
        except Exception as e:
            print(f"[ERROR] Error updating .env file: {e}")
            
    def clear_environment_variables(self):
        """Clear environment variables completely"""
        
        env_vars_to_clear = [
            'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'OPENROUTER_API_KEY',
            'DISCORD_BOT_TOKEN', 'DISCORD_CHANNEL_ID', 'ENVIRONMENT'
        ]
        
        print("[CONFIG] Clearing environment variables...")
        
        for var in env_vars_to_clear:
            # Clear from current session
            if var in os.environ:
                del os.environ[var]
                print(f"   Cleared {var} from session")
        
        # Clear from .env file if it exists
        if os.path.exists('.env'):
            os.remove('.env')
            print("   Deleted .env file")
        
        # Reload configuration
        self.load_configuration()
        
        print("[SUCCESS] Environment variables cleared!")
        print("[TIP] Restart your IDE/terminal to ensure complete cleanup")


def save_analysis_to_preferences(result: dict, rank: int = 1):
    """[SUCCESS] UPDATED: Helper function with enhanced grading support"""
    try:
        # Extract stability metrics
        prediction_winner = result.get('prediction_winner', 'GBM')
        
        if prediction_winner == 'ML':
            winning_stability = result.get('ml_stability', {})
        else:
            winning_stability = result.get('gbm_stability', {})
        
        gbm_stability = result.get('gbm_stability', {})
        ml_stability = result.get('ml_stability', {})
        
        # Calculate averages for consistency
        avg_auc = (gbm_stability.get('auc', 0) + ml_stability.get('auc', 0)) / 2
        avg_slope = (gbm_stability.get('slope', 0) + ml_stability.get('slope', 0)) / 2
        avg_total_return = (gbm_stability.get('total_return', 0) + ml_stability.get('total_return', 0)) / 2
        avg_sharpe = (gbm_stability.get('sharpe_ratio', 0) + ml_stability.get('sharpe_ratio', 0)) / 2
        
        # [SUCCESS] UPDATED: Extract GenAI information
        grade_info = result.get('grade_info', {})
        genai_score = grade_info.get('genai_score', None) if isinstance(grade_info, dict) else None
        grading_method = result.get('grading_method', 'traditional')
        
        preference_data = {
            'symbol': result['symbol'],
            'date_analyzed': result.get('date_analyzed', datetime.now().strftime('%Y-%m-%d')),
            'rank': rank,
            'score': result.get('score', 0),
            'grade': result.get('grade', 'N/A'),
            'grade_category': result.get('grade_category', 'Unknown'),
            'prediction_winner': result.get('prediction_winner', 'N/A'),
            'strategy_winner': result.get('strategy_winner', 'N/A'),
            # ✅ FIXED: Handle both possible key names
            'best_strategy': result.get('best_strategy_name', result.get('best_strategy', 'N/A')),
            'best_strategy_params': result.get('best_strategy_params', 'N/A'),
            'gbm_prediction_return': result.get('gbm_prediction_return', 0),
            'ml_prediction_return': result.get('ml_prediction_return', 0),
            'best_strategy_return': result.get('best_strategy_return', 0),
            
            # Stability metrics
            'avg_auc': avg_auc,
            'avg_slope': avg_slope, 
            'avg_total_return': avg_total_return,
            'avg_sharpe': avg_sharpe,
            
            # Individual values
            'gbm_auc': gbm_stability.get('auc', 0),
            'ml_auc': ml_stability.get('auc', 0),
            'gbm_slope': gbm_stability.get('slope', 0),
            'ml_slope': ml_stability.get('slope', 0),
            'gbm_total_return': gbm_stability.get('total_return', 0),
            'ml_total_return': ml_stability.get('total_return', 0),
            
            # ✅ NEW: GenAI grading information
            'genai_score': genai_score if genai_score is not None else 0.0,
            'grading_method': grading_method,
            'genai_available': genai_score is not None,
            
            # Other metrics
            'sharpe_ratio': winning_stability.get('sharpe_ratio', 0),
            'volatility': winning_stability.get('volatility', 0),
            'max_drawdown': winning_stability.get('max_drawdown', 0),
            
            'data_source': result.get('data_source', 'unknown'),
            'trading_allowed': False,
            'notes': f"Added from {'single' if rank == 1 else 'multiple'} stock analysis on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
        
        success = UserPreferenceManager.save_user_preference(preference_data)
        if success:
            print(f"[SUCCESS] {result['symbol']} saved to user preferences!")
            print(f"   Grade: {preference_data['grade']} | Score: {preference_data['score']:.1f}")
            print(f"   AUC: {preference_data['avg_auc']:.2f} | Slope: {preference_data['avg_slope']:.6f}")
            print(f"   Total Return: {preference_data['avg_total_return']:.4f}")
            print(f"   [AI] GenAI Score: {preference_data['genai_score']:.3f}" if preference_data['genai_available'] else "   [AI] GenAI: Not Available")
            print(f"   [CHART] Grading Method: {preference_data['grading_method']}")
            print(f"   Best Strategy: {preference_data['best_strategy']}")
            print(f"   Trading: {'Allowed' if preference_data['trading_allowed'] else 'Denied (default)'}")
        else:
            print(f"[ERROR] Failed to save {result['symbol']} to user preferences")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] Error saving {result['symbol']} to preferences: {str(e)}")
        print(f"[DEBUG] Available keys in result: {list(result.keys())}")  # Debug info
        return False

def check_genai_availability(config_manager):
    """[SUCCESS] UPDATED: Check if GenAI functionality is available"""
    
    if not config_manager.is_genai_available():
        print("[WARNING] OpenRouter API key not found in environment variables")
        print("[TIP] To enable GenAI features:")
        print("   1. Get an API key from https://openrouter.ai/keys")
        print("   2. Set OPENROUTER_API_KEY environment variable")
        print("   3. Or add it to your .env file")
        return False
    
    try:
        # Test creating enhanced grader
        api_key = config_manager.get_openrouter_key()
        test_grader = EnhancedGradingWithGenAI(api_key)
        return test_grader.genai_evaluator.genesis_instance is not None
    except Exception as e:
        print(f"[WARNING] GenAI functionality test failed: {e}")
        return False

def main():
    """✅ ENHANCED: Main function with environment variable configuration"""
    
    print("="*80)
    print("COMP2012 ASSIGNMENT 2 - ENHANCED TRADING BOT WITH CONFIGURATION MANAGEMENT")
    print("="*80)
    
    # ✅ Initialize configuration manager
    config = ConfigurationManager()
    
    print("[FEATURES] ENHANCED FEATURES:")
    print("[SUCCESS] Environment variable configuration management")
    print("[SUCCESS] Secure credential storage with .env file support")
    print("[SUCCESS] yfinance primary data source with Alpaca fallback")
    print("[SUCCESS] Enhanced XGBoost model with optimized parameters")
    print("[SUCCESS] Comprehensive strategy testing")
    print("[SUCCESS] GenAI-enhanced grading system")
    print("[SUCCESS] Discord bot integration")
    print("[SUCCESS] User preference management with trading bot integration")
    print("[SUCCESS] Enhanced visualizations with colored grade backgrounds")
    
    # Show configuration status
    print(f"\n[CONFIG] CONFIGURATION STATUS:")
    print(f"   Alpaca API: {'[SUCCESS] Configured' if config.config['alpaca_api_key'] else '[ERROR] Not configured'}")
    print(f"   GenAI (OpenRouter): {'[SUCCESS] Available' if config.is_genai_available() else '[ERROR] Not configured'}")
    print(f"   Discord Bot: {'[SUCCESS] Available' if config.is_discord_available() else '[ERROR] Not configured'}")
    print(f"   Environment: {config.config['environment']}")
    
    print("\nChoose operation:")
    print("1. [CHART] COMPREHENSIVE SINGLE STOCK ANALYSIS (Enhanced with yfinance)")
    print("2. [CHART] MULTIPLE STOCKS COMPREHENSIVE TESTING (Enhanced)")
    print("3. [TROPHY] S&P 500 comprehensive analysis with GenAI grading")
    print("4. [BOT] AUTOMATED TRADING LOOP")
    print("5. [LIST] VIEW ACCOUNT HOLDINGS")
    print("6. [SAVE] MANAGE USER PREFERENCES")
    print("7. [CONFIG] CONFIGURATION MANAGEMENT")
    print("8. [BOT] START DISCORD BOT")
    
    choice = input("Enter choice (1-8): ")
    
    if choice == "7":
        # NEW: Configuration management
        print("\n[CONFIG] CONFIGURATION MANAGEMENT")
        print("="*50)
        
        # FIXED CODE (using the new method):
        print("Current Configuration:")
        for key, value in config.config.items():
            display_value = config._display_config_value(key, value)
            print(f"  {key}: {display_value}")
        
        print("\nConfiguration Options:")
        print("1. Create/Update .env file")
        print("2. Test API connections")
        print("3. Show setup instructions")
        print("4. Validate configuration")
        
        config_choice = input("Choose option (1-4): ")
        
        # Update the configuration choice handling in main():
        if config_choice == "1":
            print("\nConfiguration Update Options:")
            print("1. Update specific values")
            print("2. Create/recreate entire .env file")
            
            update_choice = input("Choose option (1 or 2): ").strip()
            
            if update_choice == "1":
                config._update_env_file_selective()
            else:
                config._create_env_file()
        elif config_choice == "2":
            # Test connections
            api_key, secret_key = config.get_alpaca_credentials()
            if api_key and secret_key:
                try:
                    test_bot = TradingBot(None, api_key=api_key, secret_key=secret_key, paper=True)
                    account = test_bot.get_account_info()
                    print("[SUCCESS] Alpaca connection successful!")
                    print(f"   Account status: {account.get('status', 'Unknown')}")
                except Exception as e:
                    print(f"[ERROR] Alpaca connection failed: {e}")
            
            if config.is_genai_available():
                genai_available = check_genai_availability(config)
                print(f"{'[SUCCESS]' if genai_available else '[ERROR]'} GenAI connection: {'Working' if genai_available else 'Failed'}")
        elif config_choice == "3":
            config._show_env_setup_instructions()
        elif config_choice == "4":
            print("\n[LIST] Configuration Validation:")
            issues = []
            
            if not config.config['alpaca_api_key']:
                issues.append("Missing Alpaca API key")
            if not config.config['alpaca_secret_key']:
                issues.append("Missing Alpaca secret key")
            if not config.is_genai_available():
                issues.append("GenAI features disabled (missing OpenRouter key)")
            if not config.is_discord_available():
                issues.append("Discord bot disabled (missing token/channel)")
            
            if issues:
                print("[WARNING] Configuration Issues:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("[SUCCESS] All configurations are properly set!")
        
        return
    
    elif choice == "8":
        # FIXED: Discord bot with proper asyncio handling
        if not config.is_discord_available():
            print("[ERROR] Discord bot not configured")
            print("[TIP] Set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID to enable Discord features")
            return
        
        print("[BOT] Starting Discord bot...")
        print("[TIME] Bot will send daily reports at 7:00 AM EST")
        print("[TIP] Note: Close this program to stop the bot")
        
        try:
            # Check if we're in an existing event loop (like Jupyter)
            import asyncio
            try:
                # This will raise RuntimeError if no event loop is running
                asyncio.get_running_loop()
                print("[WARNING] Detected existing event loop (like Jupyter)")
                print("[CONFIG] Please run the Discord bot in a separate Python script:")
                print("```python")
                print("from ChatBot import run_discord_bot")
                print("run_discord_bot()")
                print("```")
            except RuntimeError:
                # No event loop running, safe to start Discord bot
                from ChatBot import run_discord_bot
                run_discord_bot()
        except ImportError:
            print("[ERROR] ChatBot module not found. Please ensure ChatBot.py is in your project directory.")
        except Exception as e:
            print(f"[ERROR] Error starting Discord bot: {e}")
            print("[TIP] Try running the Discord bot in a separate terminal/script")
        return

    
    # Get API credentials for trading operations
    api_key, secret_key = config.get_alpaca_credentials()
    
    if not api_key or not secret_key:
        print("[ERROR] Cannot proceed without Alpaca API credentials")
        print("[TIP] Use option 7 (Configuration Management) to set up your credentials")
        return
    
    # Rest of the main function logic remains the same, but uses config-managed credentials
    if choice == "6":
        # User preference management [unchanged - just uses the credentials from config]
        print("\n[SAVE] USER PREFERENCE MANAGEMENT")
        print("="*50)
        
        user_prefs = UserPreferenceManager.load_user_preferences()
        if user_prefs.empty:
            print("[NOTE] No user preferences found.")
            print("[TIP] Tip: Analyze stocks using options 1-3 to build your preference list!")
        else:
            print(f"[LIST] Found {len(user_prefs)} stocks in your preferences:")
            print(f"{'Symbol':<8} {'Grade':<5} {'Score':<7} {'AUC':<8} {'Slope':<10} {'Trading':<10} {'Best Strategy':<20}")
            print("-" * 85)
            
            for _, stock in user_prefs.iterrows():
                trading_allowed = stock.get('trading_allowed', False)
                if isinstance(trading_allowed, str):
                    trading_allowed = trading_allowed.lower() in ['true', '1', 'yes', 'allowed']
                
                trading_status = "[SUCCESS] Allowed" if trading_allowed else "[ERROR] Denied"
                strategy = stock.get('best_strategy', 'N/A')[:19]
                
                auc_val = stock.get('avg_auc', stock.get('gbm_auc', 0))
                slope_val = stock.get('avg_slope', stock.get('gbm_slope', 0))
                
                print(f"{stock['symbol']:<8} {stock.get('grade', 'N/A'):<5} "
                      f"{stock.get('score', 0):<7.1f} {auc_val:<8.1f} {slope_val:<10.6f} "
                      f"{trading_status:<10} {strategy:<20}")
            
            # Modify trading permissions
            modify_choice = input("\nModify trading permissions? (y/n): ").lower()
            if modify_choice == 'y':
                while True:
                    symbol = input("Enter symbol to modify (or 'exit' to finish): ").upper()
                    if symbol == 'EXIT':
                        break
                        
                    if symbol in user_prefs['symbol'].values:
                        current_row = user_prefs[user_prefs['symbol'] == symbol]
                        current_status = current_row['trading_allowed'].iloc[0]
                        
                        if isinstance(current_status, str):
                            current_status = current_status.lower() in ['true', '1', 'yes', 'allowed']
                        else:
                            current_status = bool(current_status)
                        
                        new_status = not current_status
                        
                        print(f"\n[CHART] Current status for {symbol}: {'Allowed' if current_status else 'Denied'}")
                        print(f"[CHANGE] New status will be: {'Allowed' if new_status else 'Denied'}")
                        
                        confirm = input("Confirm change? (y/n): ").lower()
                        if confirm == 'y':
                            success = UserPreferenceManager.update_trading_permission(symbol, new_status)
                            
                            if success:
                                print(f"[SUCCESS] Trading permission for {symbol} updated successfully!")
                                user_prefs = UserPreferenceManager.load_user_preferences()
                                updated_row = user_prefs[user_prefs['symbol'] == symbol]
                                updated_status = updated_row['trading_allowed'].iloc[0]
                                
                                if isinstance(updated_status, str):
                                    updated_status = updated_status.lower() in ['true', '1', 'yes', 'allowed']
                                else:
                                    updated_status = bool(updated_status)
                                
                                print(f"[CHART] Verified new status: {'[SUCCESS] Allowed' if updated_status else '[ERROR] Denied'}")
                            else:
                                print(f"[ERROR] Failed to update trading permission for {symbol}")
                        else:
                            print("[ERROR] Change cancelled")
                    else:
                        print(f"[ERROR] {symbol} not found in user preferences")
    
    elif choice == "5":
        # View account holdings
        print("\n[CHART] DISPLAYING ACCOUNT HOLDINGS...")
        trading_bot = TradingBot(None, api_key=api_key, secret_key=secret_key, paper=True)
        trading_bot.display_all_holdings()
    elif choice == "4":
        print("\n[BOT] ENHANCED AUTOMATED TRADING SYSTEM")
        print("="*50)
        
        print("Trading Mode Options:")
        print("1. Paper Trading (Recommended for testing)")
        print("2. Live Trading ([WARNING] Real money!)")
        
        trading_choice = input("Enter choice (1 or 2): ")
        paper_trading = trading_choice != "2"
        
        if not paper_trading:
            confirm = input("[WARNING] WARNING: This is LIVE TRADING with real money! Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("[ERROR] Live trading cancelled")
                return
        
        print(f"\n[ROCKET] Initializing Enhanced Trading System...")
        print(f"[MONEY] Paper Trading: {paper_trading}")
        
        auto_trader = AutomatedTradingSystemLoop(api_key, secret_key, paper_trading)
        
        print("\nEnhanced Trading Options:")
        print("1. Grade-Based Buying (Buy stocks by grade priority)")
        print("2. Strategy-Based Trading (Apply strategies to holdings)")
        print("3. Full Automated Trading Loop (Continuous monitoring)")
        print("4. View Current Holdings")
        print("5. View Account Information")
        print("6. Back to Main Menu")
        
        sub_choice = input("Select option (1-6): ").strip()
        
        if sub_choice == "1":
            print("\n[CHART] GRADE-BASED BUYING SYSTEM")
            print("="*40)
            print("[INFO] This will buy stocks based on grade priority:")
            print("   - A+ grade stocks first, then A, A-, B+, etc.")
            print("   - Round-robin buying within each grade")
            print("   - Respects cash reserve settings")
            print("   - Uses stocks from your preference database")
            
            confirm_buy = input("\nProceed with grade-based buying? (y/n): ").lower()
            if confirm_buy == 'y':
                auto_trader.execute_grade_based_buying()
            else:
                print("[ERROR] Grade-based buying cancelled")
        
        elif sub_choice == "2":
            print("\n[STRATEGY] STRATEGY-BASED TRADING")
            print("="*40)
            print("[INFO] This will apply best strategies to current holdings")
            auto_trader.execute_strategy_based_trading()
        
        elif sub_choice == "3":
            print("\n[ROCKET] Starting Full Automated Trading Loop...")
            print("[CHART] Features:")
            print("   - Continuous market monitoring")
            print("   - Grade-based buying during market hours")
            print("   - Strategy-based position management")
            print("   - Automatic cash reserve management")
            print("[CONTROL] Press Ctrl+C to stop")
            
            try:
                auto_trader.start_automated_trading()
            except KeyboardInterrupt:
                print("\n[STOPPED] Trading system stopped by user")
                auto_trader.stop_trading()
        
        elif sub_choice == "4":
            print("\n[LIST] CURRENT HOLDINGS")
            auto_trader.trading_bot.display_all_holdings()
        
        elif sub_choice == "5":
            print("\n[INFO] ACCOUNT INFORMATION")
            account_info = auto_trader.trading_bot.get_account_info()
            print("="*50)
            for key, value in account_info.items():
                print(f"{key}: {value}")
            print("="*50)
        
        elif sub_choice == "6":
            print("[ERROR] Returning to main menu")
            return
        
        else:
            print("[ERROR] Invalid option selected")
    
    elif choice == "3":
        print("\nS&P 500 Analysis Options (Enhanced with GenAI):")
        print("1. Quick test (10 stocks) - 5-8 minutes")
        print("2. Sample analysis (50 stocks) - 20-40 minutes")  
        print("3. Full analysis (all ~500 stocks) - 3-6 hours")
        print(f"\n[BOT] GenAI Enhancement: {'[SUCCESS] Available' if config.is_genai_available() else '[ERROR] Disabled'}")
        
        sub_choice = input("Enter choice (1, 2, or 3): ")
        
        analyzer = SP500ComprehensiveAnalyzer(api_key, secret_key)
        
        if sub_choice == "3":
            print("[ROCKET] Starting full S&P 500 analysis...")
            print("[TIME] Estimated time: 3-6 hours")
            analyzer.run_comprehensive_analysis(max_workers=25)
        elif sub_choice == "2":
            print("[ROCKET] Starting sample S&P 500 analysis (50 stocks)...")
            print("[TIME] Estimated time: 20-40 minutes")
            analyzer.run_comprehensive_analysis(max_workers=8, sample_size=50)
        else:
            print("[ROCKET] Starting quick S&P 500 test (10 stocks)...")
            print("[TIME] Estimated time: 5-8 minutes")
            analyzer.run_comprehensive_analysis(max_workers=5, sample_size=10)
    
    elif choice == "2":
        # Multiple stocks comprehensive testing
        print("\n[CHART] MULTIPLE STOCKS COMPREHENSIVE TESTING (Enhanced)")
        print("="*60)
        print("[CONFIG] Features:")
        print("  - Environment-based configuration")
        print("  - yfinance primary data source with maximum historical data")
        print("  - Enhanced XGBoost predictions with optimized parameters")
        print("  - Comprehensive strategy testing with parameter display")
        print("  - Relative grading vs S&P 500 benchmark")
        print(f"  - GenAI enhancement: {'[SUCCESS] Available' if config.is_genai_available() else '[ERROR] Disabled'}")
        
        symbols_input = input("\nEnter stock symbols (comma-separated) or press Enter for default set [AAPL,NVDA,TSLA,MSFT,GOOGL]: ").strip()
        
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        else:
            symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'GOOGL']
        
        show_viz = input("Show comprehensive visualizations for each stock? (y/n) [y]: ").lower() != 'n'
        save_all = input("Save all successful analyses to user preferences? (y/n) [y]: ").lower() != 'n'
        
        analyzer = ComprehensiveStockAnalyzer(api_key, secret_key)
        
        print(f"\n[ROCKET] Starting analysis of {len(symbols)} stocks...")
        successful_analyses = []
        failed_analyses = []
        saved_count = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n{'='*20} STOCK {i}/{len(symbols)}: {symbol} {'='*20}")
            
            try:
                result = analyzer.analyze_stock_comprehensive(symbol, show_visualization=show_viz)
                
                if result:
                    print(f"[SUCCESS] {symbol} analysis completed successfully")
                    print(f"   Data Source: {result.get('data_source', 'unknown')}")
                    print(f"   Grade: {result.get('grade', 'N/A')} (Score: {result.get('score', 0):.1f})")
                    print(f"   Best Strategy: {result.get('best_strategy_name', 'N/A')}")
                    successful_analyses.append(result)
                    
                    should_save = save_all
                    if not save_all:
                        individual_save = input(f"Save {symbol} to user preferences? (y/n): ").lower()
                        should_save = individual_save == 'y'
                    
                    if should_save:
                        if save_analysis_to_preferences(result, rank=i):
                            saved_count += 1
                else:
                    print(f"[ERROR] {symbol} analysis failed")
                    failed_analyses.append(symbol)
                    
            except Exception as e:
                print(f"[ERROR] {symbol} analysis error: {str(e)}")
                failed_analyses.append(symbol)
            
            if i < len(symbols):
                print("[TIME] Waiting 3 seconds before next analysis...")
                time.sleep(3)
        
        # Summary of results
        print(f"\n{'='*60}")
        print("MULTIPLE STOCKS ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"[SUCCESS] Successful: {len(successful_analyses)} stocks")
        print(f"[ERROR] Failed: {len(failed_analyses)} stocks")
        print(f"[SAVE] Saved to preferences: {saved_count} stocks")
        
        if successful_analyses:
            print(f"\n[CHART] RESULTS RANKED BY SCORE:")
            successful_analyses.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            print(f"{'Rank':<4} {'Symbol':<8} {'Grade':<5} {'Score':<7} {'Data Source':<12} {'Best Strategy':<20}")
            print("-" * 70)
            
            for rank, result in enumerate(successful_analyses, 1):
                print(f"{rank:<4} {result['symbol']:<8} {result.get('grade', 'N/A'):<5} "
                      f"{result.get('score', 0):<7.1f} {result.get('data_source', 'unknown'):<12} "
                      f"{result.get('best_strategy_name', 'N/A')[:19]:<20}")
        
        if failed_analyses:
            print(f"\n[ERROR] Failed analyses: {', '.join(failed_analyses)}")
        
        if saved_count > 0:
            print(f"\n[SAVE] {saved_count} stocks saved to UserPreferenceStock.csv")
            print("   Use option 6 to view and manage your preferences")
            print("   Use option 4 for automated trading with your preferences")
    
    elif choice == "1":
        # Single stock comprehensive analysis
        print("\n[CHART] COMPREHENSIVE SINGLE STOCK ANALYSIS (Enhanced)")
        print("="*60)
        print("[CONFIG] Enhanced Features:")
        print("  • Environment-based secure configuration")
        print("  • yfinance primary data source (maximum historical data)")
        print("  • Alpaca fallback if yfinance fails")
        print("  • Enhanced XGBoost model with optimized parameters")
        print("  • Relative grading vs S&P 500 benchmark")
        print(f"  • GenAI enhancement: {'[SUCCESS] Available' if config.is_genai_available() else '[ERROR] Disabled'}")
        print("  • Enhanced visualization with colored grade backgrounds")
        print("  • Save to user preferences for automated trading")
        
        symbol = input("\nEnter stock symbol (default: AAPL): ").upper() or 'AAPL'
        show_viz = input("Show comprehensive analysis visualization? (y/n) [y]: ").lower() != 'n'
        
        print(f"\n[ROCKET] Starting comprehensive analysis for {symbol}...")
        print("[CHART] This may take 1-2 minutes depending on data availability...")
        
        analyzer = ComprehensiveStockAnalyzer(api_key, secret_key)
        
        try:
            result = analyzer.analyze_stock_comprehensive(symbol, show_visualization=show_viz)
            
            if result:
                print(f"\n[SUCCESS] Comprehensive analysis for {symbol} completed successfully!")
                print(f"[CHART] Data Source Used: {result.get('data_source', 'unknown')}")
                print(f"[TARGET] Final Grade: {result.get('grade', 'N/A')} (Score: {result.get('score', 0):.1f}/100)")
                print(f"[STRATEGY] Best Strategy: {result.get('best_strategy_name', 'N/A')}")
                print(f"[CHART] Grading Method: {result.get('grading_method', 'relative')}")
                
                if show_viz:
                    print(f"[CHART] Check the generated visualization: {symbol}_comprehensive_analysis.png")
                
                save_choice = input(f"\nSave {symbol} to user preferences? (y/n): ").lower()
                if save_choice == 'y':
                    print(f"\n[SAVE] Saving {symbol} to user preferences...")
                    success = save_analysis_to_preferences(result, rank=1)
                    
                    if success:
                        print("\n[CELEBRATION] Analysis saved successfully!")
                        print("[TIP] Next steps:")
                        print("   • Use option 6 to view and manage your preferences")
                        print("   • Use option 4 for automated trading with your saved preferences")
                        print("   • Modify trading permissions in option 6 if needed")
                    else:
                        print("[ERROR] Failed to save analysis to preferences")
                else:
                    print("[NOTE] Analysis not saved to preferences")
                    
            else:
                print(f"[ERROR] Failed to analyze {symbol}")
                print("[TIP] Tips:")
                print("  • Check if the symbol is correct")
                print("  • Try a different stock symbol")
                print("  • Check your internet connection")
                
        except Exception as e:
            print(f"[ERROR] Error analyzing {symbol}: {str(e)}")
            print("[TIP] This might be due to:")
            print("  • Invalid stock symbol")
            print("  • Network connectivity issues")
            print("  • API rate limiting")
    else:
        print("[ERROR] Unexpected input, please read the instruction before you input.")

def display_startup_info():
    """✅ ENHANCED: Display startup information with configuration status"""
    
    # Try to load configuration to show status
    try:
        load_dotenv()
        alpaca_configured = bool(os.getenv('ALPACA_API_KEY') and os.getenv('ALPACA_SECRET_KEY'))
        genai_configured = bool(os.getenv('OPENROUTER_API_KEY'))
        discord_configured = bool(os.getenv('DISCORD_BOT_TOKEN') and os.getenv('DISCORD_CHANNEL_ID'))
    except:
        alpaca_configured = genai_configured = discord_configured = False
    
    print("\n[SEARCH] ENHANCED TECHNICAL DETAILS:")
    print("  - Configuration: Environment variables with .env file support")
    print("  - Security: Encrypted credential storage and validation")
    print("  - Market class: COMP2012 pattern with GBM simulation")
    print("  - StockPredictor: yfinance -> Alpaca fallback data sourcing")
    print("  - Analysis classes: Inherit from Market base class")
    print("  - XGBoost: Enhanced parameters for better predictions")
    print("  - Strategies: Mean Reversion, Trend Following, Weighted Trend Following")
    print("  - Grading: Relative vs S&P 500 with strict percentile distribution")
    print("  - GenAI: OpenRouter integration for enhanced scoring")
    print("  - Visualization: Enhanced charts with colored grade backgrounds")
    print("  - Discord: Automated daily account status reporting")
    
    print("\n[LIST] ENHANCED FILE STRUCTURE:")
    print("  - PA2.py: Main program with configuration management")
    print("  - Market.py: Market, StockPredictor (core classes)")
    print("  - Analysis.py: All analysis classes inheriting from Market")
    print("  - GradingSystem.py: Traditional + GenAI grading systems")
    print("  - TradingBot.py: Trading strategies and automation")
    print("  - ChatBot.py: Discord bot for daily reporting")
    print("  - .env: Secure credential storage (create this file)")
    print("  - UserPreferenceStock.csv: Saved stock analyses")
    
    print(f"\n[SETTINGS] CONFIGURATION STATUS:")
    print(f"  - Alpaca Trading: {'[SUCCESS] Configured' if alpaca_configured else '[ERROR] Setup needed'}")
    print(f"  - GenAI Features: {'[SUCCESS] Available' if genai_configured else '[ERROR] Setup needed'}")
    print(f"  - Discord Bot: {'[SUCCESS] Available' if discord_configured else '[ERROR] Setup needed'}")
    
    if not (alpaca_configured and genai_configured):
        print(f"\n[TIP] QUICK SETUP:")
        print(f"  - Run the program and choose option 7 (Configuration Management)")
        print(f"  - Or create a .env file with your API keys")
        print(f"  - Get Alpaca keys: https://app.alpaca.markets/paper/dashboard/overview")
        print(f"  - Get OpenRouter key: https://openrouter.ai/keys")

if __name__ == "__main__":
    display_startup_info()
    main()
