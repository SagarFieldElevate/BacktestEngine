"""
Base template for trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class StrategyTemplate(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        self.parameters = parameters or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signal(self, market_state: Dict[str, Any], 
                       portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signal based on market state and portfolio
        
        Args:
            market_state: Current market data and indicators
            portfolio: Current portfolio state
            
        Returns:
            Dictionary with:
                - action: 'BUY', 'SELL', or 'HOLD'
                - size: Position size (for BUY/SELL)
                - confidence: Signal confidence (0-1)
                - reason: Optional explanation
        """
        pass
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        return True
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required market indicators"""
        return []


class SimpleMovingAverageCrossover(StrategyTemplate):
    """Example strategy: SMA crossover"""
    
    def __init__(self):
        super().__init__({
            'fast_period': 20,
            'slow_period': 50,
            'position_size': 0.95
        })
        
    def generate_signal(self, market_state: Dict[str, Any], 
                       portfolio: Dict[str, Any]) -> Dict[str, Any]:
        
        fast_sma = market_state.get(f'sma_{self.parameters["fast_period"]}', 0)
        slow_sma = market_state.get(f'sma_{self.parameters["slow_period"]}', 0)
        
        if fast_sma == 0 or slow_sma == 0:
            return {'action': 'HOLD'}
        
        # Check for golden cross (bullish)
        if fast_sma > slow_sma and portfolio['bitcoin'] == 0:
            return {
                'action': 'BUY',
                'size': (portfolio['cash'] * self.parameters['position_size']) / market_state['close'],
                'confidence': 0.7,
                'reason': f'Golden Cross: SMA{self.parameters["fast_period"]} > SMA{self.parameters["slow_period"]}'
            }
        
        # Check for death cross (bearish)
        elif fast_sma < slow_sma and portfolio['bitcoin'] > 0:
            return {
                'action': 'SELL',
                'size': portfolio['bitcoin'],
                'confidence': 0.7,
                'reason': f'Death Cross: SMA{self.parameters["fast_period"]} < SMA{self.parameters["slow_period"]}'
            }
        
        return {'action': 'HOLD'}
    
    def get_required_indicators(self) -> List[str]:
        return [f'sma_{self.parameters["fast_period"]}', 
                f'sma_{self.parameters["slow_period"]}']


class RSIMeanReversion(StrategyTemplate):
    """Example strategy: RSI mean reversion"""
    
    def __init__(self):
        super().__init__({
            'rsi_period': 14,
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'position_size': 0.5
        })
        
    def generate_signal(self, market_state: Dict[str, Any], 
                       portfolio: Dict[str, Any]) -> Dict[str, Any]:
        
        rsi = market_state.get(f'rsi_{self.parameters["rsi_period"]}', 50)
        
        # Buy when oversold
        if rsi < self.parameters['oversold_threshold'] and portfolio['bitcoin'] == 0:
            confidence = (self.parameters['oversold_threshold'] - rsi) / self.parameters['oversold_threshold']
            return {
                'action': 'BUY',
                'size': (portfolio['cash'] * self.parameters['position_size']) / market_state['close'],
                'confidence': min(confidence, 0.9),
                'reason': f'RSI oversold: {rsi:.2f}'
            }
        
        # Sell when overbought
        elif rsi > self.parameters['overbought_threshold'] and portfolio['bitcoin'] > 0:
            confidence = (rsi - self.parameters['overbought_threshold']) / (100 - self.parameters['overbought_threshold'])
            return {
                'action': 'SELL',
                'size': portfolio['bitcoin'],
                'confidence': min(confidence, 0.9),
                'reason': f'RSI overbought: {rsi:.2f}'
            }
        
        return {'action': 'HOLD'}
    
    def get_required_indicators(self) -> List[str]:
        return [f'rsi_{self.parameters["rsi_period"]}']