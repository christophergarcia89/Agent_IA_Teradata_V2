"""
Azure OpenAI utilities for the Teradata SQL Agent.
"""

import os
import logging
from typing import Optional
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from config.settings import settings

logger = logging.getLogger(__name__)


class AzureOpenAIManager:
    """Manager for Azure OpenAI client configurations."""
    
    def __init__(self):
        self._langchain_client = None
        self._openai_client = None
        self._initialized = False
        
    def test_connection(self) -> bool:
        """Test connection to Azure OpenAI."""
        try:
            client = self.get_langchain_client()
            # Hacer una llamada simple para probar la conexión
            response = client.invoke("Test connection to Azure OpenAI")
            return True
        except Exception as e:
            logger.error(f"Error testing Azure OpenAI connection: {e}")
            return False
            
    def get_langchain_client(self) -> AzureChatOpenAI:
        """Get LangChain Azure OpenAI client."""
        if not self._langchain_client:
            self._langchain_client = AzureChatOpenAI(
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_deployment_name,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                temperature=0.1,
                max_tokens=1500,
                timeout=30.0
            )
            logger.info("Azure OpenAI LangChain client configured: %s", settings.azure_openai_deployment_name)
        return self._langchain_client
    
    def get_openai_client(self) -> AzureOpenAI:
        """Get native Azure OpenAI client."""
        if not self._openai_client:
            self._openai_client = AzureOpenAI(
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_deployment_name,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                timeout=30.0
            )
            logger.info("Azure OpenAI client configured: %s", settings.azure_openai_deployment_name)
        return self._openai_client
    
    def get_llm(self) -> AzureChatOpenAI:
        """Get LLM client (alias for get_langchain_client for compatibility)."""
        return self.get_langchain_client()


# Global instance
azure_openai_manager = AzureOpenAIManager()


def get_azure_openai_client() -> AzureOpenAI:
    """Get the configured Azure OpenAI client."""
    return azure_openai_manager.get_openai_client()


def get_azure_langchain_client() -> AzureChatOpenAI:
    """Get the configured LangChain Azure OpenAI client."""
    return azure_openai_manager.get_langchain_client()


def get_llm() -> AzureChatOpenAI:
    """Get LLM client (alias for get_azure_langchain_client for compatibility)."""
    return azure_openai_manager.get_llm()


def test_azure_connection() -> bool:
    """Test the Azure OpenAI connection."""
    try:
        client = get_azure_openai_client()
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )
        logger.info("Azure OpenAI connection test successful")
        return True
    except Exception as e:
        logger.error(f"Azure OpenAI connection test failed: {e}")
        return False
