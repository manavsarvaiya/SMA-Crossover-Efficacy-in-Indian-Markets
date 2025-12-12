import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import scipy.stats as stats
from scipy.stats import skew, kurtosis, shapiro, jarque_bera
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NiftyStrategyAnalyzer:
    """
    SMA Strategy Analysis Framework for NIFTY 50 Minute Data
    """
    
    def __init__(self, data_file='NIFTY 50_minute_data.csv', transaction_cost=0.00015):
        """
        Initialize the strategy analyzer
        
        Parameters:
        data_file: str, path to the CSV file
        transaction_cost: float, transaction cost per trade (0.015%)
        """
        self.data_file = data_file
        self.transaction_cost = transaction_cost
        self.sma_periods = [5, 10, 20, 50, 100, 200]
        self.strategy_results = {}
        self.optimal_sma = None
        self.market_hours_only = True
        
        # Annualization factors - MATCHING ORIGINAL
        self.minutes_per_day = 375  # 9:15 to 15:30 = 6.25 hours = 375 minutes
        self.trading_days_per_year = 250
        self.annualization_factor = np.sqrt(self.trading_days_per_year * self.minutes_per_day)
        self.return_annualization_factor = self.trading_days_per_year * self.minutes_per_day
        
    def load_data(self):
        """Load and prepare the NIFTY 50 minute data"""
        print("Processing NIFTY 50 minute data...")
        
        df = pd.read_csv(self.data_file, parse_dates=True)
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H.%M')
        df.set_index('DateTime', inplace=True)
        
        # Filter to market hours if specified
        if self.market_hours_only:
            df = df.between_time('09:15', '15:30')
            
        self.df = df
        print(f"Data processed: {len(self.df)} records from {self.df.index[0]} to {self.df.index[-1]}")
        print(f"Data dimensions: {self.df.shape}")
        
        return self.df
    
    def data_investigation(self):
        """Comprehensive data investigation"""
        print("\n" + "=" * 60)
        print("DATA INVESTIGATION")
        print("=" * 60)
        
        print("\nData Summary:")
        print("-" * 30)
        print(self.df.describe())
        
        print(f"\nMissing Data Check:")
        print("-" * 30)
        print(self.df.isnull().sum())
        
        # Temporal features
        self.df['hour'] = self.df.index.hour
        self.df['minute'] = self.df.index.minute
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['time_of_day'] = (self.df.index.hour - 9) * 60 + self.df.index.minute - 15
        
        # Return calculations
        self.df['returns'] = self.df['Close'].pct_change()
        self.df['log_returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Volatility estimation
        self.df['volatility_5'] = self.df['returns'].rolling(5).std()
        self.df['volatility_20'] = self.df['returns'].rolling(20).std()
        
        # Price relationships
        self.df['high_low_ratio'] = self.df['High'] / self.df['Low']
        self.df['open_close_ratio'] = self.df['Open'] / self.df['Close']
        
        print(f"\nReturn Characteristics:")
        print("-" * 30)
        returns_data = self.df['returns'].dropna()
        print(f"Average Return: {returns_data.mean():.6f}")
        print(f"Return Std Dev: {returns_data.std():.6f}")
        print(f"Return Skewness: {stats.skew(returns_data):.4f}")
        print(f"Return Kurtosis: {stats.kurtosis(returns_data):.4f}")
        
        # Distribution assessment
        normality_test = stats.shapiro(returns_data.sample(5000))
        print(f"Shapiro-Wilk p-value: {normality_test[1]:.2e}")
        print(f"Returns are {'NOT ' if normality_test[1] < 0.05 else ''}normally distributed")
        
        # ARCH test
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_result = het_arch(returns_data.dropna(), nlags=5)
            print(f"\nARCH Test p-value: {arch_result[1]:.4f}")
            print(f"Returns {'exhibit' if arch_result[1] < 0.05 else 'do not exhibit'} ARCH effects")
        except:
            print("\nARCH test could not be performed")
        
        self._generate_investigation_visuals()
        
        return self.df
    
    def _generate_investigation_visuals(self):
        """Generate data investigation visuals"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Price progression
        plt.subplot(2, 2, 1)
        plt.plot(self.df.index, self.df['Close'], linewidth=0.7, color='navy')
        plt.title('NIFTY 50 Price Movement', fontsize=14, fontweight='bold')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # 2. Return distribution
        plt.subplot(2, 2, 2)
        clean_returns = self.df['returns'].dropna()
        plt.hist(clean_returns, bins=80, alpha=0.7, color='green', density=True)
        plt.axvline(clean_returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {clean_returns.mean():.6f}')
        plt.axvline(clean_returns.median(), color='orange', linestyle='--',
                   label=f'Median: {clean_returns.median():.6f}')
        plt.title('Return Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Daily pattern analysis
        plt.subplot(2, 2, 3)
        hour_volatility = self.df.groupby('hour')['returns'].std()
        plt.plot(hour_volatility.index, hour_volatility.values, marker='o', 
                linewidth=2, markersize=6, color='red')
        plt.title('Hourly Volatility Pattern', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Volatility')
        plt.grid(True, alpha=0.3)
        
        # 4. Day of week effect - FIXED: Handle missing days
        plt.subplot(2, 2, 4)
        daily_returns = self.df.groupby('day_of_week')['returns'].mean()
        
        # Create complete range for days 0-6 (Monday to Sunday)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_returns_complete = pd.Series(index=range(7), dtype=float)
        for day in range(7):
            if day in daily_returns.index:
                daily_returns_complete[day] = daily_returns[day]
            else:
                daily_returns_complete[day] = 0  # No data for this day
        
        plt.bar(range(len(days)), daily_returns_complete.values, color='purple', alpha=0.7)
        plt.title('Day of Week Return Pattern', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Return')
        plt.xticks(range(len(days)), days)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_investigation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_features(self):
        """Create analytical features"""
        print(f"\nFeature Generation")
        print("-" * 30)
        
        # Moving average calculations
        for period in self.sma_periods:
            self.df[f'SMA_{period}'] = self.df['Close'].rolling(period).mean()
            self.df[f'price_above_sma_{period}'] = (self.df['Close'] > self.df[f'SMA_{period}']).astype(int)
            self.df[f'sma_slope_{period}'] = self.df[f'SMA_{period}'].diff()
            
        # Distance from SMA
        for period in self.sma_periods:
            self.df[f'dist_from_sma_{period}'] = (self.df['Close'] - self.df[f'SMA_{period}']) / self.df[f'SMA_{period}']
        
        # SMA crossovers
        for i, period in enumerate(self.sma_periods[:-1]):
            fast_sma = f'SMA_{period}'
            slow_sma = f'SMA_{self.sma_periods[i + 1]}'
            self.df[f'{fast_sma}_above_{slow_sma}'] = (self.df[fast_sma] > self.df[slow_sma]).astype(int)
        
        # Momentum indicators
        self.df['momentum_5'] = self.df['Close'] / self.df['Close'].shift(5) - 1
        self.df['momentum_20'] = self.df['Close'] / self.df['Close'].shift(20) - 1
        
        # Volume indicators
        self.df['volume_sma_20'] = self.df['volume'].rolling(20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']
        
        print(f"Generated features for {len(self.sma_periods)} SMA periods")
        print(f"Total features: {self.df.shape[1]}")
        
        return self.df
    
    def evaluate_single_ma(self, ma_period):
        """Evaluate single moving average strategy - MATCHING ORIGINAL"""
        if 'returns' not in self.df.columns:
            self.df['returns'] = self.df['Close'].pct_change()
            
        sma_col = f'SMA_{ma_period}'
        signal_col = f'signal_{ma_period}'
        
        # Generate trading signals - MATCHING ORIGINAL
        self.df[signal_col] = np.where(self.df['Close'] > self.df[sma_col], 1, 0)
        
        # Calculate position changes
        self.df[f'position_change_{ma_period}'] = self.df[signal_col].diff()
        
        # Calculate strategy returns
        self.df[f'strategy_returns_{ma_period}'] = self.df[signal_col].shift(1) * self.df['returns']
        
        # Account for transaction costs - MATCHING ORIGINAL
        transaction_costs = abs(self.df[f'position_change_{ma_period}']) * self.transaction_cost
        self.df[f'net_returns_{ma_period}'] = self.df[f'strategy_returns_{ma_period}'] - transaction_costs
        
        # Calculate cumulative returns
        self.df[f'cum_returns_{ma_period}'] = (1 + self.df[f'net_returns_{ma_period}']).cumprod()
        
        performance_metrics = self._calculate_strategy_metrics(ma_period)
        return performance_metrics
    
    def _calculate_strategy_metrics(self, ma_period):
        """Calculate comprehensive strategy metrics - MATCHING ORIGINAL"""
        returns_col = f'net_returns_{ma_period}'
        cum_returns_col = f'cum_returns_{ma_period}'
        signal_col = f'signal_{ma_period}'
        
        strategy_returns = self.df[returns_col].dropna()
        cum_returns = self.df[cum_returns_col].dropna()
        signals = self.df[signal_col].dropna()
        
        if len(strategy_returns) == 0:
            return {
                'MA_Period': ma_period,
                'Total_Return': 0,
                'Annualized_Return': 0,
                'Volatility': 0,
                'Sharpe': 0,
                'Max_Drawdown': 0,
                'Win_Rate': 0,
                'Profit_Factor': 0,
                'Calmar': 0,
                'Sortino': 0,
                'Trade_Count': 0,
                'Avg_Trade': 0,
                'Best_Trade': 0,
                'Worst_Trade': 0,
                'Time_in_Market': 0
            }
        
        # Total return - MATCHING ORIGINAL
        total_return = cum_returns.iloc[-1] - 1
        
        # Annualized return - MATCHING ORIGINAL
        n_periods = len(strategy_returns)
        n_years = n_periods / self.return_annualization_factor
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Volatility (annualized) - MATCHING ORIGINAL
        volatility = strategy_returns.std() * self.annualization_factor
        
        # Sharpe ratio - MATCHING ORIGINAL
        sharpe = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown - MATCHING ORIGINAL
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate - MATCHING ORIGINAL
        winning_trades = strategy_returns[strategy_returns > 0]
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Trade statistics - MATCHING ORIGINAL
        position_changes = self.df[f'position_change_{ma_period}'].abs().sum()
        n_trades = int(position_changes / 2)  # Round trips
        
        # Profit factor - MATCHING ORIGINAL
        gross_profit = winning_trades.sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average trade - MATCHING ORIGINAL
        avg_trade = strategy_returns[strategy_returns != 0].mean()
        
        # Best and worst trades - MATCHING ORIGINAL
        best_trade = strategy_returns.max()
        worst_trade = strategy_returns.min()
        
        # Calmar ratio - MATCHING ORIGINAL
        calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Sortino ratio - MATCHING ORIGINAL
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * self.annualization_factor
        sortino = annualized_return / downside_deviation if downside_deviation > 0 else np.inf
        
        # Time in market - MATCHING ORIGINAL
        time_in_market = signals.mean()
        
        metrics = {
            'MA_Period': ma_period,
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe': sharpe,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Calmar': calmar,
            'Sortino': sortino,
            'Trade_Count': n_trades,
            'Avg_Trade': avg_trade,
            'Best_Trade': best_trade,
            'Worst_Trade': worst_trade,
            'Time_in_Market': time_in_market
        }
        
        return metrics
    
    def execute_backtest(self):
        """Execute backtest for all moving average periods"""
        print(f"\nStrategy Backtesting")
        print("-" * 30)
        all_metrics = []
        
        for ma_period in self.sma_periods:
            print(f"Evaluating SMA({ma_period})...")
            period_metrics = self.evaluate_single_ma(ma_period)
            all_metrics.append(period_metrics)
            self.strategy_results[ma_period] = period_metrics
            
        self.results_summary = pd.DataFrame(all_metrics)
        self.optimal_sma = self.results_summary.loc[self.results_summary['Sharpe'].idxmax(), 'MA_Period']
        
        print(f"\nBacktesting completed")
        print(f"Most effective SMA: {self.optimal_sma} (Sharpe: {self.results_summary['Sharpe'].max():.3f})")
        
        return self.results_summary
    
    def _generate_detailed_analysis(self):
        """Generate detailed analysis plots"""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Hourly performance for best SMA
        plt.subplot(2, 2, 1)
        best_returns_col = f'net_returns_{self.optimal_sma}'
        if best_returns_col in self.df.columns:
            hourly_perf = self.df.groupby('hour')[best_returns_col].mean()
            plt.bar(hourly_perf.index, hourly_perf.values * 100, color='steelblue', alpha=0.7)
            plt.title(f'Hourly Performance - SMA({self.optimal_sma})', fontsize=12, fontweight='bold')
            plt.xlabel('Hour of Day')
            plt.ylabel('Average Return (%)')
            plt.grid(True, alpha=0.3)
        
        # 2. Signal distribution
        plt.subplot(2, 2, 2)
        signal_col = f'signal_{self.optimal_sma}'
        if signal_col in self.df.columns:
            signal_counts = self.df[signal_col].value_counts()
            plt.pie(signal_counts.values, labels=['Out of Market', 'In Market'], 
                   autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen'])
            plt.title(f'Market Position Distribution - SMA({self.optimal_sma})', fontsize=12, fontweight='bold')
        
        # 3. Rolling Sharpe ratio
        plt.subplot(2, 2, 3)
        if best_returns_col in self.df.columns:
            rolling_sharpe = self.df[best_returns_col].rolling(252*375).mean() / self.df[best_returns_col].rolling(252*375).std() * np.sqrt(252*375)
            plt.plot(self.df.index, rolling_sharpe, linewidth=1, color='darkgreen')
            plt.title(f'Rolling Sharpe Ratio (1-year) - SMA({self.optimal_sma})', fontsize=12, fontweight='bold')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True, alpha=0.3)
        
        # 4. Trade timing analysis
        plt.subplot(2, 2, 4)
        if 'position_change_' + str(self.optimal_sma) in self.df.columns:
            trades_by_hour = self.df[self.df[f'position_change_{self.optimal_sma}'] != 0].groupby('hour').size()
            plt.bar(trades_by_hour.index, trades_by_hour.values, color='orange', alpha=0.7)
            plt.title(f'Trade Timing by Hour - SMA({self.optimal_sma})', fontsize=12, fontweight='bold')
            plt.xlabel('Hour of Day')
            plt.ylabel('Number of Trades')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_performance(self):
        """Generate performance visualizations"""
        print(f"\nPerformance Visualization")
        print("-" * 30)
        
        # Performance overview
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Cumulative returns comparison
        plt.subplot(2, 3, 1)
        for ma_period in self.sma_periods:
            cum_col = f'cum_returns_{ma_period}'
            if cum_col in self.df.columns:
                plt.plot(self.df.index, self.df[cum_col], label=f'SMA({ma_period})', linewidth=1.2)
        
        # Benchmark comparison
        benchmark_returns = (1 + self.df['returns']).cumprod()
        plt.plot(self.df.index, benchmark_returns, label='Buy & Hold', 
                linewidth=2, linestyle='--', color='black')
        
        plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Cumulative Return')
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 2. Drawdown analysis
        plt.subplot(2, 3, 2)
        best_cum_col = f'cum_returns_{self.optimal_sma}'
        if best_cum_col in self.df.columns:
            cum_returns = self.df[best_cum_col].dropna()
            rolling_max = cum_returns.expanding().max()
            drawdown = (cum_returns - rolling_max) / rolling_max
            plt.fill_between(self.df.index[:len(drawdown)], drawdown * 100, 0, alpha=0.7, color='red')
            plt.title(f'Drawdown - Best SMA({self.optimal_sma})', fontsize=14, fontweight='bold')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
        
        # 3. Performance metrics
        plt.subplot(2, 3, 3)
        metrics_to_plot = ['Sharpe', 'Calmar', 'Win_Rate']
        x = np.arange(len(self.sma_periods))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = self.results_summary[metric].values
            plt.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        plt.xlabel('SMA Period')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x + width, self.sma_periods)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Risk-return profile
        plt.subplot(2, 3, 4)
        scatter = plt.scatter(self.results_summary['Volatility'] * 100, 
                            self.results_summary['Annualized_Return'] * 100,
                            s=100, alpha=0.7, 
                            c=self.results_summary['Sharpe'], 
                            cmap='viridis')
        
        for i, sma in enumerate(self.sma_periods):
            plt.annotate(f'SMA({sma})', 
                        (self.results_summary.iloc[i]['Volatility'] * 100, 
                         self.results_summary.iloc[i]['Annualized_Return'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Volatility (Annualized %)')
        plt.ylabel('Return (Annualized %)')
        plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.grid(True, alpha=0.3)
        
        # 5. Maximum drawdown comparison
        plt.subplot(2, 3, 5)
        mdd_values = [abs(x) * 100 for x in self.results_summary['Max_Drawdown'].values]
        colors = ['red' if x == max(mdd_values) else 'lightblue' for x in mdd_values]
        bars = plt.bar(range(len(self.sma_periods)), mdd_values, color=colors, alpha=0.7)
        
        plt.xlabel('SMA Period')
        plt.ylabel('Maximum Drawdown (%)')
        plt.title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(self.sma_periods)), [f'SMA({p})' for p in self.sma_periods])
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, mdd_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Trade analysis
        plt.subplot(2, 3, 6)
        trade_metrics = ['Trade_Count', 'Win_Rate', 'Profit_Factor']
        normalized_data = self.results_summary[trade_metrics].copy()
        
        for col in trade_metrics:
            if normalized_data[col].max() > 0:
                normalized_data[col] = normalized_data[col] / normalized_data[col].max()
        
        x = np.arange(len(self.sma_periods))
        width = 0.25
        
        for i, metric in enumerate(trade_metrics):
            plt.bar(x + i * width, normalized_data[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('SMA Period')
        plt.ylabel('Normalized Value')
        plt.title('Trade Metrics (Normalized)', fontsize=14, fontweight='bold')
        plt.xticks(x + width, self.sma_periods)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed analysis
        self._generate_detailed_analysis()
    
    def statistical_analysis_best_sma(self):
        """Perform detailed statistical analysis of the best SMA strategy - MATCHING ORIGINAL"""
        print(f"\nStatistical Analysis - Best SMA({self.optimal_sma})")
        print("-" * 50)
        
        returns_col = f'net_returns_{self.optimal_sma}'
        returns = self.df[returns_col].dropna()
        returns_trading = returns[returns != 0]  # Only trading periods
        
        if len(returns_trading) == 0:
            print("No trading returns available for analysis")
            return
        
        print("RETURN STATISTICS:")
        print(f"Mean return: {returns_trading.mean():.6f} ({returns_trading.mean() * self.return_annualization_factor * 100:.2f}% annualized)")
        print(f"Median return: {returns_trading.median():.6f}")
        print(f"Std deviation: {returns_trading.std():.6f} ({returns_trading.std() * self.annualization_factor * 100:.2f}% annualized)")
        print(f"Skewness: {skew(returns_trading):.4f}")
        print(f"Kurtosis: {kurtosis(returns_trading):.4f}")
        print(f"Min return: {returns_trading.min():.6f} ({returns_trading.min() * 100:.3f}%)")
        print(f"Max return: {returns_trading.max():.6f} ({returns_trading.max() * 100:.3f}%)")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\nPERCENTILES:")
        for p in percentiles:
            value = np.percentile(returns_trading, p)
            print(f"{p:2d}th percentile: {value:.6f} ({value * 100:.3f}%)")
        
        # Statistical tests
        print(f"\nSTATISTICAL TESTS:")
        
        # Normality test
        if len(returns_trading) > 5000:
            sample_returns = returns_trading.sample(5000)
        else:
            sample_returns = returns_trading
        
        shapiro_stat, shapiro_p = shapiro(sample_returns)
        print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.2e}")
        print(f"Returns are {'NOT ' if shapiro_p < 0.05 else ''}normally distributed")
        
        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(returns_trading)
        print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_p:.2e}")
        
        # One-sample t-test
        t_stat, t_p = stats.ttest_1samp(returns_trading, 0)
        print(f"T-test (mean=0): statistic={t_stat:.4f}, p-value={t_p:.2e}")
        print(f"Mean return is {'significantly' if t_p < 0.05 else 'not significantly'} different from 0")
        
        # Win/Loss analysis - MATCHING ORIGINAL FORMAT
        print(f"\nWIN/LOSS PATTERN ANALYSIS:")
        print("-" * 30)
        
        wins = returns_trading[returns_trading > 0]
        losses = returns_trading[returns_trading < 0]
        
        print(f"Winning trades: {len(wins)} ({len(wins)/len(returns_trading)*100:.1f}%)")
        print(f"Losing trades: {len(losses)} ({len(losses)/len(returns_trading)*100:.1f}%)")
        print(f"Neutral trades: {len(returns_trading) - len(wins) - len(losses)}")
        
        print(f"\nWINNING TRADES:")
        print(f"  Average: {wins.mean():.6f} ({wins.mean() * 100:.3f}%)")
        print(f"  Median: {wins.median():.6f}")
        print(f"  Best: {wins.max():.6f} ({wins.max() * 100:.3f}%)")
        print(f"  Total contribution: {wins.sum():.6f}")
        
        print(f"\nLOSING TRADES:")
        print(f"  Average: {losses.mean():.6f} ({losses.mean() * 100:.3f}%)")
        print(f"  Median: {losses.median():.6f}")
        print(f"  Worst: {losses.min():.6f} ({losses.min() * 100:.3f}%)")
        print(f"  Total contribution: {losses.sum():.6f}")
        
        # Streak analysis - MATCHING ORIGINAL
        win_signals = (returns_trading > 0).astype(int)
        streaks = []
        current_streak = 0
        streak_type = None
        
        for signal in win_signals:
            if signal == 1:  # Win
                if streak_type == 'win':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(('loss', current_streak))
                    current_streak = 1
                    streak_type = 'win'
            else:  # Loss
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(('win', current_streak))
                    current_streak = 1
                    streak_type = 'loss'
        
        # Add final streak
        if current_streak > 0:
            streaks.append((streak_type, current_streak))
        
        win_streaks = [s[1] for s in streaks if s[0] == 'win']
        loss_streaks = [s[1] for s in streaks if s[0] == 'loss']
        
        print(f"\nSTREAK ANALYSIS:")
        if win_streaks:
            print(f"Win streaks - Max: {max(win_streaks)}, Avg: {np.mean(win_streaks):.1f}")
        if loss_streaks:
            print(f"Loss streaks - Max: {max(loss_streaks)}, Avg: {np.mean(loss_streaks):.1f}")
    
    def evaluate_crossover_strategies(self):
        """Evaluate dual moving average crossover strategies - MATCHING ORIGINAL"""
        print("\n" + "=" * 80)
        print("DUAL SMA CROSSOVER ANALYSIS")
        print("=" * 80)
        
        fast_periods = [5, 10, 20, 50]
        slow_periods = [20, 50, 100, 200]
        crossover_results = []
        self.crossover_equity = {}
        self.crossover_trades = {}
        
        strategy_count = sum(1 for fast in fast_periods for slow in slow_periods if fast < slow)
        print(f"Testing {strategy_count} dual SMA combinations...")
        
        for fast in fast_periods:
            for slow in slow_periods:
                if fast < slow:
                    combo_name = f"{fast}/{slow}"
                    print(f"  → Analyzing {combo_name} combination...")
                    
                    # Create dual SMA strategy signals - MATCHING ORIGINAL
                    fast_sma = self.df[f'SMA_{fast}']
                    slow_sma = self.df[f'SMA_{slow}']
                    
                    # Signal: Long when fast SMA > slow SMA - MATCHING ORIGINAL
                    signal = (fast_sma > slow_sma).astype(int)
                    
                    # Calculate returns with transaction costs - MATCHING ORIGINAL
                    strategy_returns = signal.shift(1) * self.df['returns']
                    position_changes = signal.diff().abs()
                    transaction_costs = position_changes * self.transaction_cost
                    net_returns = strategy_returns - transaction_costs
                    
                    # Store data for detailed analysis
                    self.crossover_equity[combo_name] = (1 + net_returns.dropna()).cumprod()
                    trade_returns = net_returns[position_changes == 1].dropna()
                    self.crossover_trades[combo_name] = trade_returns
                    
                    # Calculate comprehensive metrics - MATCHING ORIGINAL
                    cum_returns = self.crossover_equity[combo_name]
                    if len(cum_returns) == 0:
                        continue
                    
                    total_return = cum_returns.iloc[-1] - 1
                    volatility = net_returns.std() * self.annualization_factor
                    
                    # Sharpe ratio - MATCHING ORIGINAL
                    if net_returns.std() > 0:
                        sharpe = (net_returns.mean() / net_returns.std()) * self.annualization_factor
                    else:
                        sharpe = 0
                    
                    # Drawdown analysis - MATCHING ORIGINAL
                    rolling_max = cum_returns.expanding().max()
                    drawdown = (cum_returns - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    # Trade analysis - MATCHING ORIGINAL
                    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
                    avg_win = trade_returns[trade_returns > 0].mean() if len(trade_returns[trade_returns > 0]) > 0 else 0
                    avg_loss = trade_returns[trade_returns < 0].mean() if len(trade_returns[trade_returns < 0]) > 0 else 0
                    
                    # Profit factor - MATCHING ORIGINAL FORMULA
                    profit_factor = -avg_win / avg_loss * win_rate / (1 - win_rate) if avg_loss < 0 and win_rate < 1 else np.inf
                    
                    # Annualized return - MATCHING ORIGINAL
                    n_periods = len(net_returns.dropna())
                    n_years = n_periods / self.return_annualization_factor
                    annualized_return = (1 + total_return) ** (1 / n_years) - 1
                    
                    # Risk-adjusted metrics - MATCHING ORIGINAL
                    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
                    downside_returns = net_returns[net_returns < 0]
                    
                    if len(downside_returns) > 0:
                        sortino_ratio = net_returns.mean() / downside_returns.std() * self.annualization_factor
                    else:
                        sortino_ratio = np.inf
                    
                    # Distribution analysis - MATCHING ORIGINAL
                    if len(trade_returns) > 2:
                        trade_skew = skew(trade_returns)
                        trade_kurt = kurtosis(trade_returns)
                        _, shapiro_p = shapiro(trade_returns) if len(trade_returns) >= 3 else (np.nan, np.nan)
                        _, jb_p = jarque_bera(trade_returns) if len(trade_returns) >= 3 else (np.nan, np.nan)
                    else:
                        trade_skew = trade_kurt = shapiro_p = jb_p = np.nan
                    
                    # Add to results - MATCHING ORIGINAL STRUCTURE
                    crossover_results.append({
                        'Fast_SMA': fast,
                        'Slow_SMA': slow,
                        'Strategy': combo_name,
                        'Total_Return': total_return,
                        'Annualized_Return': annualized_return,
                        'Volatility': volatility,
                        'Sharpe_Ratio': sharpe,
                        'Calmar_Ratio': calmar_ratio,
                        'Sortino_Ratio': sortino_ratio,
                        'Max_Drawdown': max_drawdown,
                        'Win_Rate': win_rate,
                        'Trade_Count': len(trade_returns),
                        'Avg_Win': avg_win,
                        'Avg_Loss': avg_loss,
                        'Profit_Factor': profit_factor,
                        'Trade_Skew': trade_skew,
                        'Trade_Kurtosis': trade_kurt,
                        'Shapiro_P': shapiro_p,
                        'JB_P': jb_p
                    })
        
        # Create results dataframe and sort by Sharpe ratio
        self.crossover_summary = pd.DataFrame(crossover_results)
        self.crossover_summary = self.crossover_summary.sort_values('Sharpe_Ratio', ascending=False)
        self.best_crossover = self.crossover_summary.iloc[0]['Strategy']
        
        print(f"\nAnalysis complete! Best strategy: {self.best_crossover}")
        print(f"  Sharpe Ratio: {self.crossover_summary.iloc[0]['Sharpe_Ratio']:.3f}")
        print(f"  Total Return: {self.crossover_summary.iloc[0]['Total_Return'] * 100:.2f}%")
        
        return self.crossover_summary
    
    def generate_strategy_report(self):
        """Generate comprehensive strategy report - MATCHING ORIGINAL FORMAT"""
        print("\n" + "=" * 80)
        print("DETAILED DUAL SMA STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Top crossover strategies - MATCHING ORIGINAL
        print("\nTOP 10 DUAL SMA COMBINATIONS (Ranked by Sharpe Ratio):")
        print("-" * 100)
        display_cols = ['Strategy', 'Total_Return', 'Annualized_Return', 'Sharpe_Ratio',
                       'Calmar_Ratio', 'Max_Drawdown', 'Win_Rate', 'Trade_Count', 'Profit_Factor']
        display_df = self.crossover_summary[display_cols].head(10).copy()
        
        # Format for better readability - MATCHING ORIGINAL
        for col in ['Total_Return', 'Annualized_Return', 'Max_Drawdown', 'Win_Rate']:
            display_df[col] = (display_df[col] * 100).round(2)
        for col in ['Sharpe_Ratio', 'Calmar_Ratio', 'Profit_Factor']:
            display_df[col] = display_df[col].round(3)
        
        print(display_df.to_string(index=False))
        
        # Best strategy detailed analysis - MATCHING ORIGINAL FORMAT
        best_strategy_row = self.crossover_summary.iloc[0]
        best_trades = self.crossover_trades[self.best_crossover]
        
        print(f"\n\nDETAILED ANALYSIS - BEST STRATEGY: {self.best_crossover}")
        print("=" * 70)
        
        # Performance metrics - MATCHING ORIGINAL FORMAT
        print(f"PERFORMANCE METRICS:")
        print(f"├─ Total Return: {best_strategy_row['Total_Return'] * 100:.2f}%")
        print(f"├─ Annualized Return: {best_strategy_row['Annualized_Return'] * 100:.2f}%")
        print(f"├─ Volatility: {best_strategy_row['Volatility'] * 100:.2f}%")
        print(f"├─ Sharpe Ratio: {best_strategy_row['Sharpe_Ratio']:.3f}")
        print(f"├─ Calmar Ratio: {best_strategy_row['Calmar_Ratio']:.3f}")
        print(f"├─ Sortino Ratio: {best_strategy_row['Sortino_Ratio']:.3f}")
        print(f"└─ Maximum Drawdown: {best_strategy_row['Max_Drawdown'] * 100:.2f}%")
        
        # Trade statistics - MATCHING ORIGINAL FORMAT
        print(f"\nTRADE STATISTICS:")
        print(f"├─ Total Trades: {best_strategy_row['Trade_Count']}")
        print(f"├─ Win Rate: {best_strategy_row['Win_Rate'] * 100:.2f}%")
        print(f"├─ Average Winning Trade: {best_strategy_row['Avg_Win'] * 100:.3f}%")
        print(f"├─ Average Losing Trade: {best_strategy_row['Avg_Loss'] * 100:.3f}%")
        print(f"├─ Profit Factor: {best_strategy_row['Profit_Factor']:.3f}")
        if best_strategy_row['Avg_Loss'] < 0:
            print(f"├─ Win/Loss Ratio: {-best_strategy_row['Avg_Win'] / best_strategy_row['Avg_Loss']:.3f}")
        
        # Distribution statistics - MATCHING ORIGINAL FORMAT
        print(f"\nRETURN DISTRIBUTION STATISTICS:")
        print(f"├─ Mean Trade Return: {best_trades.mean() * 100:.4f}%")
        print(f"├─ Median Trade Return: {best_trades.median() * 100:.4f}%")
        print(f"├─ Standard Deviation: {best_trades.std() * 100:.4f}%")
        print(f"├─ Skewness: {best_strategy_row['Trade_Skew']:.3f}")
        print(f"├─ Kurtosis: {best_strategy_row['Trade_Kurtosis']:.3f}")
        print(f"├─ Shapiro-Wilk p-value: {best_strategy_row['Shapiro_P']:.4f}")
        print(f"└─ Jarque-Bera p-value: {best_strategy_row['JB_P']:.4f}")
        
        # Percentile analysis - MATCHING ORIGINAL FORMAT
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\nTRADE RETURN PERCENTILES:")
        for p in percentiles:
            pct_val = np.percentile(best_trades, p)
            print(f"├─ {p:2d}th percentile: {pct_val * 100:.3f}%")
        
        # Market regime analysis - MATCHING ORIGINAL FORMAT
        print(f"\nMARKET REGIME INSIGHTS:")
        best_equity = self.crossover_equity[self.best_crossover]
        monthly_returns = best_equity.resample('M').last().pct_change().dropna()
        if len(monthly_returns) > 0:
            positive_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)
            print(f"├─ Positive months: {positive_months}/{total_months} ({positive_months/total_months*100:.1f}%)")
            print(f"├─ Best month: {monthly_returns.max() * 100:.2f}%")
            print(f"├─ Worst month: {monthly_returns.min() * 100:.2f}%")
            print(f"└─ Monthly volatility: {monthly_returns.std() * 100:.2f}%")
        
        return best_strategy_row, best_trades
    
    def analyze_parameter_characteristics(self):
        """Analyze parameter characteristics - MATCHING ORIGINAL FORMAT"""
        print(f"\nPARAMETER CHARACTERISTIC ANALYSIS")
        print("=" * 50)
        
        # Fast MA analysis - MATCHING ORIGINAL
        fast_analysis = self.crossover_summary.groupby('Fast_SMA').agg({
            'Sharpe_Ratio': 'mean',
            'Total_Return': 'mean',
            'Win_Rate': 'mean'
        }).round(3)
        
        # Slow MA analysis - MATCHING ORIGINAL
        slow_analysis = self.crossover_summary.groupby('Slow_SMA').agg({
            'Sharpe_Ratio': 'mean',
            'Total_Return': 'mean',
            'Win_Rate': 'mean'
        }).round(3)
        
        print(f"\nFAST SMA ANALYSIS:")
        print(fast_analysis)
        
        print(f"\nSLOW SMA ANALYSIS:")
        print(slow_analysis)
        
        optimal_fast = fast_analysis['Sharpe_Ratio'].idxmax()
        optimal_slow = slow_analysis['Sharpe_Ratio'].idxmax()
        
        print(f"\nOPTIMAL PARAMETER OBSERVATIONS:")
        print(f"├─ Best Fast SMA Period: {optimal_fast}")
        print(f"└─ Best Slow SMA Period: {optimal_slow}")
        
        return fast_analysis, slow_analysis

def execute_analysis():
    """Execute complete analysis workflow"""
    analyzer = NiftyStrategyAnalyzer()
    
    # Data processing
    df = analyzer.load_data()
    analyzer.data_investigation()
    df = analyzer.generate_features()
    
    # Single SMA analysis
    analyzer.execute_backtest()
    analyzer.visualize_performance()
    analyzer.statistical_analysis_best_sma()
    
    # Crossover analysis
    analyzer.evaluate_crossover_strategies()
    analyzer.generate_strategy_report()
    analyzer.analyze_parameter_characteristics()
    
    print("\nAnalysis workflow completed successfully!")

if __name__ == "__main__":
    execute_analysis()