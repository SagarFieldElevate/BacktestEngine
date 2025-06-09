"""
Airtable helper utilities
"""

import os
import logging
from typing import List, Dict, Any, Optional
from airtable import Airtable
import pandas as pd
import config

logger = logging.getLogger(__name__)


class AirtableHelper:
    def __init__(self):
        self.api_key = config.AIRTABLE_API_KEY
        self.base_id = config.AIRTABLE_BASE_ID
        
        if not self.api_key or not self.base_id:
            raise ValueError("Airtable API key and base ID must be configured")
        
    def get_table(self, table_name: str) -> Airtable:
        """Get Airtable instance for specific table"""
        return Airtable(self.base_id, table_name, self.api_key)
    
    def fetch_records(self, table_name: str, 
                     fields: Optional[List[str]] = None,
                     formula: Optional[str] = None,
                     sort: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch records from Airtable"""
        try:
            table = self.get_table(table_name)
            
            params = {}
            if fields:
                params['fields'] = fields
            if formula:
                params['formula'] = formula
            if sort:
                params['sort'] = sort
            
            records = []
            for page in table.get_iter(**params):
                for record in page:
                    records.append(record['fields'])
            
            logger.info(f"Fetched {len(records)} records from {table_name}")
            return records
            
        except Exception as e:
            logger.error(f"Error fetching records from {table_name}: {e}")
            raise
    
    def fetch_excel_attachment(self, table_name: str, 
                             record_id: str,
                             attachment_field: str = 'Excel File') -> pd.DataFrame:
        """Fetch Excel file attachment from Airtable record"""
        try:
            table = self.get_table(table_name)
            record = table.get(record_id)
            
            if attachment_field not in record['fields']:
                raise ValueError(f"No attachment found in field {attachment_field}")
            
            attachments = record['fields'][attachment_field]
            if not attachments:
                raise ValueError("No attachments found")
            
            # Get first attachment URL
            attachment_url = attachments[0]['url']
            
            # Read Excel file directly from URL
            df = pd.read_excel(attachment_url)
            
            logger.info(f"Fetched Excel file from {table_name}, record {record_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Excel attachment: {e}")
            raise
    
    def get_latest_data_version(self, table_name: str = "Data_Versions") -> Dict[str, Any]:
        """Get latest data version information"""
        try:
            records = self.fetch_records(
                table_name=table_name,
                sort=[{'field': 'Created', 'direction': 'desc'}]
            )
            
            if records:
                return records[0]
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching data version: {e}")
            return {}