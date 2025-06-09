"""
Performance metrics calculator for backtest results
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class MetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_all_metrics(self, equity_curve: pd.DataFrame, 
                            trades: pd.DataFrame, 
                            initial_capital: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        metrics = {}
        
        # Basic returns
        metrics['total_return'] = self.calculate_total_return(equity_curve, initial_capital)
        metrics['annualized_return'] = self.calculate_annualized_return(equity_curve, initial_capital)
        
        # Risk metrics
        metrics['volatility'] = self.calculate_volatility(equity_curve)
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(equity_curve, initial_capital)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(equity_curve, initial_capital)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(equity_curve, initial_capital)
        
        # Drawdown metrics
        drawdown_stats = self.calculate_drawdown_stats(equity_curve)
        metrics.update(drawdown_stats)
        
        # Trade statistics
        if not trades.empty:
            trade_stats = self.calculate_trade_statistics(trades)
            metrics.update(trade_stats)
        else:
            metrics.update({
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0
            })
        
        # Risk-adjusted metrics
        metrics['risk_adjusted_return'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        
        return metrics
    
    def calculate_total_return(self, equity_curve: pd.DataFrame, initial_capital: float) -> float:
        """Calculate total return percentage"""
        final_value = equity_curve['total_value'].iloc[-1]
        return ((final_value - initial_capital) / initial_capital) * 100
    
    def calculate_annualized_return(self, equity_curve: pd.DataFrame, initial_capital: float) -> float:
        """Calculate annualized return"""
        total_return = self.calculate_total_return(equity_curve, initial_capital) / 100
        years = len(equity_curve) / 252  # Trading days
        
        if years > 0:
            annualized = (1 + total_return) ** (1 / years) - 1
            return annualized * 100
        return 0
    
    def calculate_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        returns = equity_curve['total_value'].pct_change().dropna()
        return returns.std() * np.sqrt(252) * 100
    
    def calculate_sharpe_ratio(self, equity_curve: pd.DataFrame, initial_capital: float) -> float:
        """Calculate Sharpe ratio"""
        returns = equity_curve['total_value'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0
        
        annualized_return = self.calculate_annualized_return(equity_curve, initial_capital) / 100
        volatility = self.calculate_volatility(equity_curve) / 100
        
        if volatility > 0:
            return (annualized_return - self.risk_free_rate) / volatility
        return 0
    
    def calculate_sortino_ratio(self, equity_curve: pd.DataFrame, initial_capital: float) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        returns = equity_curve['total_value'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0
        
        annualized_return = self.calculate_annualized_return(equity_curve, initial_capital) / 100
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                return (annualized_return - self.risk_free_rate) / downside_deviation
        
        return 0
    
    def calculate_calmar_ratio(self, equity_curve: pd.DataFrame, initial_capital: float) -> float:
        """Calculate Calmar ratio (return / max drawdown)"""
        annualized_return = self.calculate_annualized_return(equity_curve, initial_capital)
        max_drawdown = self.calculate_drawdown_stats(equity_curve)['max_drawdown']
        
        if abs(max_drawdown) > 0:
            return annualized_return / abs(max_drawdown)
        return 0
    
    def calculate_drawdown_stats(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate drawdown statistics"""
        cumulative_returns = equity_curve['total_value']
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        
        stats = {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0,
            'max_drawdown_duration': self._calculate_max_drawdown_duration(drawdown)
        }
        
        return stats
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        in_drawdown = drawdown < 0
        
        if not in_drawdown.any():
            return 0
        
        # Find consecutive drawdown periods
        drawdown_periods = []
        start = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        if start is not None:
            drawdown_periods.append(len(drawdown) - start)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate detailed trade statistics"""
        if trades.empty:
            return {}
        
        # Separate buy and sell trades
        buy_trades = trades[trades['type'] == 'BUY'].copy()
        sell_trades = trades[trades['type'] == 'SELL'].copy()
        
        # Calculate returns for closed trades
        trade_returns = []
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades.iloc[i]['price']
            sell_price = sell_trades.iloc[i]['price']
            trade_return = (sell_price - buy_price) / buy_price * 100
            trade_returns.append(trade_return)
        
        if not trade_returns:
            return {
                'total_trades': len(trades),
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0,
                'avg_trade_return': 0
            }
        
        trade_returns = np.array(trade_returns)
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        stats = {
            'total_trades': len(trades),
            'closed_trades': len(trade_returns),
            'win_rate': (len(winning_trades) / len(trade_returns)) * 100 if len(trade_returns) > 0 else 0,
            'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'avg_trade_return': trade_returns.mean(),
            'best_trade': trade_returns.max(),
            'worst_trade': trade_returns.min(),
            'avg_trade_duration': self._calculate_avg_trade_duration(buy_trades, sell_trades)
        }
        
        # Calculate profit factor
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        stats['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate expectancy
        win_rate = stats['win_rate'] / 100
        stats['expectancy'] = (win_rate * stats['avg_win']) + ((1 - win_rate) * stats['avg_loss'])
        
        return stats
    
    def _calculate_avg_trade_duration(self, buy_trades: pd.DataFrame, 
                                    sell_trades: pd.DataFrame) -> float:
        """Calculate average trade duration in days"""
        durations = []
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_date = pd.to_datetime(buy_trades.iloc[i]['date'])
            sell_date = pd.to_datetime(sell_trades.iloc[i]['date'])
            duration = (sell_date - buy_date).days
            durations.append(duration)
        
        return np.mean(durations) if durations else 0