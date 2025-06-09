"""
Strategy loader from Pinecone vector database
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import pinecone
import config
from utils.pinecone_helper import PineconeHelper

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Strategy data class"""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    entry_conditions: List[Dict[str, Any]]
    exit_conditions: List[Dict[str, Any]]
    risk_management: Dict[str, Any]
    metadata: Dict[str, Any]


class StrategyLoader:
    def __init__(self):
        self.helper = PineconeHelper()
        self.index = self.helper.get_index(config.PINECONE_INDEX_NAME)
        
    def load_strategy(self, strategy_id: str) -> Strategy:
        """Load a specific strategy by ID"""
        try:
            result = self.index.fetch(ids=[strategy_id])
            
            if strategy_id not in result['vectors']:
                raise ValueError(f"Strategy {strategy_id} not found")
            
            vector_data = result['vectors'][strategy_id]
            return self._parse_strategy(strategy_id, vector_data['metadata'])
            
        except Exception as e:
            logger.error(f"Error loading strategy {strategy_id}: {e}")
            raise
    
    def load_all_strategies(self) -> List[Strategy]:
        """Load all strategies from Pinecone"""
        try:
            # Query all strategies
            results = self.index.query(
                vector=[0] * 32,  # Dummy vector matching index dimension
                top_k=1000,
                include_metadata=True
            )
            
            strategies = []
            for match in results['matches']:
                strategy = self._parse_strategy(match['id'], match['metadata'])
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            raise
    
    def load_top_strategies(self, n: int = 5) -> List[Strategy]:
        """Load top N performing strategies"""
        try:
            # Query strategies with performance metadata
            results = self.index.query(
                vector=[0] * 32,  # Dummy vector matching index dimension
                top_k=n,
                include_metadata=True,
                filter={"performance_rank": {"$lte": n}}
            )
            
            strategies = []
            for match in results['matches']:
                strategy = self._parse_strategy(match['id'], match['metadata'])
                strategies.append(strategy)
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error loading top strategies: {e}")
            raise
    
    def _parse_strategy(self, strategy_id: str, metadata: Dict[str, Any]) -> Strategy:
        """Parse strategy from Pinecone metadata"""
        return Strategy(
            id=strategy_id,
            name=metadata.get('name', f'Strategy_{strategy_id}'),
            description=metadata.get('description', ''),
            parameters=json.loads(metadata.get('parameters', '{}')),
            entry_conditions=json.loads(metadata.get('entry_conditions', '[]')),
            exit_conditions=json.loads(metadata.get('exit_conditions', '[]')),
            risk_management=json.loads(metadata.get('risk_management', '{}')),
            metadata={
                'created_at': metadata.get('created_at'),
                'updated_at': metadata.get('updated_at'),
                'performance_rank': metadata.get('performance_rank'),
                'sharpe_ratio': metadata.get('sharpe_ratio'),
                'max_drawdown': metadata.get('max_drawdown')
            }
        )
    
    def save_strategy_performance(self, strategy_id: str, performance_metrics: Dict[str, Any]):
        """Update strategy performance metrics in Pinecone"""
        try:
            # Fetch current strategy
            result = self.index.fetch(ids=[strategy_id])
            
            if strategy_id in result['vectors']:
                current_metadata = result['vectors'][strategy_id]['metadata']
                
                # Update performance metrics
                current_metadata.update({
                    'sharpe_ratio': performance_metrics['sharpe_ratio'],
                    'max_drawdown': performance_metrics['max_drawdown'],
                    'total_return': performance_metrics['total_return'],
                    'win_rate': performance_metrics['win_rate'],
                    'last_backtest': datetime.now().isoformat()
                })
                
                # Update in Pinecone
                self.index.update(
                    id=strategy_id,
                    metadata=current_metadata
                )
                
                logger.info(f"Updated performance metrics for strategy {strategy_id}")
                
        except Exception as e:
            logger.error(f"Error updating strategy performance: {e}")