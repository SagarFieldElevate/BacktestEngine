"""
Pinecone vector database helper utilities
"""

import logging
from typing import List, Dict, Any, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
import config

logger = logging.getLogger(__name__)


class PineconeHelper:
    def __init__(self):
        self.api_key = config.PINECONE_API_KEY
        
        if not self.api_key:
            raise ValueError("Pinecone API key must be configured")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
    def get_index(self, index_name: str):
        """Get or create Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                logger.info(f"Creating new index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=32,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            return self.pc.Index(index_name)
            
        except Exception as e:
            logger.error(f"Error accessing Pinecone index {index_name}: {e}")
            raise
    
    def upsert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]):
        """Upsert vectors to Pinecone index"""
        try:
            index = self.get_index(index_name)
            
            # Format vectors for upsert
            formatted_vectors = []
            for v in vectors:
                formatted_vectors.append({
                    'id': v['id'],
                    'values': v['values'],
                    'metadata': v.get('metadata', {})
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(formatted_vectors), batch_size):
                batch = formatted_vectors[i:i + batch_size]
                index.upsert(vectors=batch)
            
            logger.info(f"Upserted {len(vectors)} vectors to {index_name}")
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise
    
    def query_similar(self, index_name: str, 
                     query_vector: List[float],
                     top_k: int = 10,
                     filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query similar vectors from Pinecone"""
        try:
            index = self.get_index(index_name)
            
            query_params = {
                'vector': query_vector,
                'top_k': top_k,
                'include_metadata': True
            }
            
            if filter:
                query_params['filter'] = filter
            
            results = index.query(**query_params)
            
            return results['matches']
            
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            raise
    
    def delete_all(self, index_name: str):
        """Delete all vectors from index"""
        try:
            index = self.get_index(index_name)
            index.delete(delete_all=True)
            logger.info(f"Deleted all vectors from {index_name}")
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise