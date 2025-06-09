"""
Core backtest engine for executing trading strategies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json

from strategy.metrics import MetricsCalculator
from strategy.templates.strategy_template import StrategyTemplate
import config

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, market_data: pd.DataFrame, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Filter data by date range
        if start_date:
            self.market_data = self.market_data[self.market_data.index >= start_date]
        if end_date:
            self.market_data = self.market_data[self.market_data.index <= end_date]
        
        self.metrics_calculator = MetricsCalculator()
        
    def run_backtest(self, strategy: 'Strategy') -> Dict[str, Any]:
        """Execute backtest for a given strategy"""
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Initialize portfolio state
        portfolio = {
            'cash': self.initial_capital,
            'bitcoin': 0,
            'total_value': self.initial_capital,
            'positions': [],
            'trades': []
        }
        
        # Initialize results tracking
        equity_curve = []
        signals = []
        
        # Convert strategy to executable template
        strategy_template = self._create_strategy_template(strategy)
        
        # Iterate through each trading day
        for idx, (date, row) in enumerate(self.market_data.iterrows()):
            # Get current market state
            market_state = self._get_market_state(idx)
            
            # Generate trading signal
            signal = strategy_template.generate_signal(market_state, portfolio)
            signals.append({
                'date': date,
                'signal': signal['action'],
                'confidence': signal.get('confidence', 1.0)
            })
            
            # Execute trade if signal
            if signal['action'] != 'HOLD':
                trade_result = self._execute_trade(
                    signal, portfolio, row['Close'], date
                )
                if trade_result:
                    portfolio['trades'].append(trade_result)
            
            # Update portfolio value
            portfolio['total_value'] = portfolio['cash'] + (portfolio['bitcoin'] * row['Close'])
            
            # Record equity curve
            equity_curve.append({
                'date': date,
                'total_value': portfolio['total_value'],
                'cash': portfolio['cash'],
                'bitcoin': portfolio['bitcoin'],
                'price': row['Close']
            })
        
        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        trades_df = pd.DataFrame(portfolio['trades']) if portfolio['trades'] else pd.DataFrame()
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            equity_df, trades_df, self.initial_capital
        )
        
        return {
            'strategy_id': strategy.id,
            'strategy_name': strategy.name,
            'metrics': metrics,
            'equity_curve': equity_df,
            'trades': trades_df,
            'signals': pd.DataFrame(signals).set_index('date')
        }
    
    def _create_strategy_template(self, strategy: 'Strategy') -> StrategyTemplate:
        """Convert strategy definition to executable template"""
        
        class DynamicStrategy(StrategyTemplate):
            def __init__(self, strategy_def):
                self.strategy_def = strategy_def
                self.parameters = strategy_def.parameters
                
            def generate_signal(self, market_state: Dict, portfolio: Dict) -> Dict:
                # Evaluate entry conditions
                if self._should_enter(market_state, portfolio):
                    return {
                        'action': 'BUY',
                        'size': self._calculate_position_size(market_state, portfolio),
                        'confidence': self._calculate_confidence(market_state)
                    }
                
                # Evaluate exit conditions
                elif self._should_exit(market_state, portfolio):
                    return {
                        'action': 'SELL',
                        'size': portfolio['bitcoin'],
                        'confidence': 1.0
                    }
                
                return {'action': 'HOLD'}
            
            def _should_enter(self, market_state: Dict, portfolio: Dict) -> bool:
                if portfolio['bitcoin'] > 0:  # Already have position
                    return False
                
                for condition in self.strategy_def.entry_conditions:
                    if not self._evaluate_condition(condition, market_state):
                        return False
                return True
            
            def _should_exit(self, market_state: Dict, portfolio: Dict) -> bool:
                if portfolio['bitcoin'] == 0:  # No position to exit
                    return False
                
                # Check stop loss
                if 'stop_loss' in self.strategy_def.risk_management:
                    current_price = market_state['close']
                    avg_entry_price = self._get_avg_entry_price(portfolio)
                    loss_pct = (current_price - avg_entry_price) / avg_entry_price
                    
                    if loss_pct <= -self.strategy_def.risk_management['stop_loss']:
                        return True
                
                # Check exit conditions
                for condition in self.strategy_def.exit_conditions:
                    if self._evaluate_condition(condition, market_state):
                        return True
                
                return False
            
            def _evaluate_condition(self, condition: Dict, market_state: Dict) -> bool:
                indicator = condition['indicator']
                operator = condition['operator']
                value = condition['value']
                
                if indicator not in market_state:
                    return False
                
                current_value = market_state[indicator]
                
                if operator == '>':
                    return current_value > value
                elif operator == '<':
                    return current_value < value
                elif operator == '>=':
                    return current_value >= value
                elif operator == '<=':
                    return current_value <= value
                elif operator == '==':
                    return current_value == value
                elif operator == 'crosses_above':
                    prev_value = market_state.get(f'{indicator}_prev', current_value)
                    return prev_value <= value and current_value > value
                elif operator == 'crosses_below':
                    prev_value = market_state.get(f'{indicator}_prev', current_value)
                    return prev_value >= value and current_value < value
                
                return False
            
            def _calculate_position_size(self, market_state: Dict, portfolio: Dict) -> float:
                risk_pct = self.strategy_def.risk_management.get('position_size', 0.95)
                available_capital = portfolio['cash']
                position_value = available_capital * risk_pct
                bitcoin_amount = position_value / market_state['close']
                return bitcoin_amount
            
            def _calculate_confidence(self, market_state: Dict) -> float:
                # Simple confidence based on number of confirming indicators
                confidence = 0.5
                
                # Add confidence based on trend alignment
                if market_state.get('sma_50', 0) > market_state.get('sma_200', 0):
                    confidence += 0.2
                
                # Add confidence based on momentum
                if market_state.get('rsi_14', 50) > 50:
                    confidence += 0.15
                
                # Add confidence based on volume
                if market_state.get('volume', 0) > market_state.get('volume_avg', 0):
                    confidence += 0.15
                
                return min(confidence, 1.0)
            
            def _get_avg_entry_price(self, portfolio: Dict) -> float:
                if not portfolio['positions']:
                    return 0
                
                total_cost = sum(p['price'] * p['amount'] for p in portfolio['positions'])
                total_amount = sum(p['amount'] for p in portfolio['positions'])
                
                return total_cost / total_amount if total_amount > 0 else 0
        
        return DynamicStrategy(strategy)
    
    def _get_market_state(self, idx: int) -> Dict[str, Any]:
        """Get current market state for strategy evaluation"""
        current_row = self.market_data.iloc[idx]
        
        state = {
            'date': current_row.name,
            'open': current_row['Open'],
            'high': current_row['High'],
            'low': current_row['Low'],
            'close': current_row['Close'],
            'volume': current_row['Volume'],
            'returns': current_row.get('Returns', 0),
            'volatility': current_row.get('Volatility', 0)
        }
        
        # Add technical indicators
        for col in ['RSI_14', 'RSI_30', 'SMA_20', 'SMA_50', 'SMA_200', 
                    'EMA_12', 'EMA_26', 'DXY', 'FearGreed']:
            if col in current_row:
                state[col.lower()] = current_row[col]
        
        # Add previous values for crossover detection
        if idx > 0:
            prev_row = self.market_data.iloc[idx - 1]
            for col in ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']:
                if col in prev_row:
                    state[f'{col.lower()}_prev'] = prev_row[col]
        
        # Add volume average
        if idx >= 20:
            state['volume_avg'] = self.market_data['Volume'].iloc[idx-20:idx].mean()
        
        return state
    
    def _execute_trade(self, signal: Dict, portfolio: Dict, 
                      current_price: float, date: datetime) -> Optional[Dict]:
        """Execute trade and update portfolio"""
        
        if signal['action'] == 'BUY':
            # Calculate trade size with commission and slippage
            size = signal['size']
            execution_price = current_price * (1 + self.slippage)
            trade_value = size * execution_price
            commission_cost = trade_value * self.commission
            total_cost = trade_value + commission_cost
            
            if total_cost > portfolio['cash']:
                # Adjust size to available cash
                available_cash = portfolio['cash'] * 0.99  # Leave small buffer
                trade_value = available_cash / (1 + self.commission)
                size = trade_value / execution_price
                commission_cost = trade_value * self.commission
                total_cost = trade_value + commission_cost
            
            if size > 0:
                # Execute buy
                portfolio['cash'] -= total_cost
                portfolio['bitcoin'] += size
                portfolio['positions'].append({
                    'date': date,
                    'amount': size,
                    'price': execution_price
                })
                
                return {
                    'date': date,
                    'type': 'BUY',
                    'size': size,
                    'price': execution_price,
                    'value': trade_value,
                    'commission': commission_cost,
                    'cash_after': portfolio['cash'],
                    'bitcoin_after': portfolio['bitcoin']
                }
        
        elif signal['action'] == 'SELL':
            # Calculate trade with commission and slippage
            size = min(signal['size'], portfolio['bitcoin'])
            execution_price = current_price * (1 - self.slippage)
            trade_value = size * execution_price
            commission_cost = trade_value * self.commission
            net_proceeds = trade_value - commission_cost
            
            if size > 0:
                # Execute sell
                portfolio['cash'] += net_proceeds
                portfolio['bitcoin'] -= size
                
                # Clear positions if fully sold
                if portfolio['bitcoin'] < 0.0001:  # Small threshold for rounding
                    portfolio['bitcoin'] = 0
                    portfolio['positions'] = []
                
                return {
                    'date': date,
                    'type': 'SELL',
                    'size': size,
                    'price': execution_price,
                    'value': trade_value,
                    'commission': commission_cost,
                    'cash_after': portfolio['cash'],
                    'bitcoin_after': portfolio['bitcoin']
                }
        
        return None
    
    def generate_report(self, strategy: 'Strategy', results: Dict[str, Any]):
        """Generate individual strategy report"""
        from pathlib import Path
        import json
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create report directory
        report_dir = Path(config.REPORTS_DIR) / strategy.id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = report_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_clean = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in results['metrics'].items()
            }
            json.dump(metrics_clean, f, indent=2)
        
        # Save trades
        if not results['trades'].empty:
            trades_file = report_dir / 'trades.csv'
            results['trades'].to_csv(trades_file, index=False)
        
        # Generate equity curve plot
        plt.figure(figsize=(12, 6))
        equity_curve = results['equity_curve']
        
        plt.subplot(2, 1, 1)
        plt.plot(equity_curve.index, equity_curve['total_value'], label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        plt.title(f'{strategy.name} - Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        running_max = equity_curve['total_value'].expanding().max()
        drawdown = (equity_curve['total_value'] - running_max) / running_max * 100
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        plt.title('Drawdown %')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(report_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate trade analysis plot
        if not results['trades'].empty:
            trades_df = results['trades'].copy()
            
            # Calculate trade returns
            buy_trades = trades_df[trades_df['type'] == 'BUY'].copy()
            sell_trades = trades_df[trades_df['type'] == 'SELL'].copy()
            
            if len(sell_trades) > 0:
                plt.figure(figsize=(10, 6))
                
                # Match buy and sell trades
                trade_returns = []
                for i, sell in sell_trades.iterrows():
                    # Find corresponding buy (simplified - assumes FIFO)
                    if i < len(buy_trades):
                        buy = buy_trades.iloc[i]
                        trade_return = (sell['price'] - buy['price']) / buy['price'] * 100
                        trade_returns.append(trade_return)
                
                if trade_returns:
                    plt.subplot(2, 1, 1)
                    plt.hist(trade_returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    plt.axvline(x=0, color='red', linestyle='--')
                    plt.title('Trade Returns Distribution')
                    plt.xlabel('Return (%)')
                    plt.ylabel('Frequency')
                    
                    # Cumulative returns
                    plt.subplot(2, 1, 2)
                    cumulative_returns = np.cumprod(1 + np.array(trade_returns) / 100) - 1
                    plt.plot(cumulative_returns * 100)
                    plt.title('Cumulative Trade Returns')
                    plt.xlabel('Trade Number')
                    plt.ylabel('Cumulative Return (%)')
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(report_dir / 'trade_analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Generate summary report
        summary = {
            'strategy_id': strategy.id,
            'strategy_name': strategy.name,
            'backtest_period': {
                'start': str(equity_curve.index[0]),
                'end': str(equity_curve.index[-1]),
                'days': len(equity_curve)
            },
            'performance': results['metrics'],
            'trade_summary': {
                'total_trades': len(results['trades']),
                'buy_trades': len(results['trades'][results['trades']['type'] == 'BUY']),
                'sell_trades': len(results['trades'][results['trades']['type'] == 'SELL'])
            }
        }
        
        summary_file = report_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Report generated for {strategy.name} at {report_dir}")
    
    def generate_comparative_report(self, results: Dict[str, Dict]):
        """Generate comparative analysis across multiple strategies"""
        from pathlib import Path
        import matplotlib.pyplot as plt
        
        comparison_dir = Path(config.REPORTS_DIR) / 'comparative_analysis'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect metrics for comparison
        comparison_data = []
        for strategy_id, result in results.items():
            comparison_data.append({
                'strategy_id': strategy_id,
                'strategy_name': result['strategy_name'],
                **result['metrics']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        comparison_df.to_csv(comparison_dir / 'strategy_comparison.csv', index=False)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sharpe Ratio comparison
        ax = axes[0, 0]
        comparison_df.plot(x='strategy_name', y='sharpe_ratio', kind='bar', ax=ax)
        ax.set_title('Sharpe Ratio Comparison')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Sharpe Ratio')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        # Total Return comparison
        ax = axes[0, 1]
        comparison_df.plot(x='strategy_name', y='total_return', kind='bar', ax=ax, color='green')
        ax.set_title('Total Return Comparison')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Total Return (%)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Max Drawdown comparison
        ax = axes[1, 0]
        comparison_df.plot(x='strategy_name', y='max_drawdown', kind='bar', ax=ax, color='red')
        ax.set_title('Maximum Drawdown Comparison')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Max Drawdown (%)')
        
        # Win Rate comparison
        ax = axes[1, 1]
        comparison_df.plot(x='strategy_name', y='win_rate', kind='bar', ax=ax, color='orange')
        ax.set_title('Win Rate Comparison')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Win Rate (%)')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(comparison_dir / 'strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot all equity curves together
        plt.figure(figsize=(12, 8))
        for strategy_id, result in results.items():
            equity_curve = result['equity_curve']
            returns = (equity_curve['total_value'] / self.initial_capital - 1) * 100
            plt.plot(returns.index, returns, label=result['strategy_name'], linewidth=2)
        
        plt.title('Strategy Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.savefig(comparison_dir / 'equity_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparative analysis generated at {comparison_dir}")