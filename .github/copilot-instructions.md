# Copilot Instructions

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Context
This is a LangGraph LCEL project for Teradata SQL query review and optimization using RAG (Retrieval Augmented Generation).

## Key Components
- **RAG System**: Uses internal knowledge base with SQL standards and examples
- **Three Agents**: 
  1. SQL Query Reviewer (uses RAG for standards compliance)
  2. Query Explain Generator (uses teradata-mcp-server)
  3. Explain Interpreter (provides optimization suggestions)
- **Knowledge Base**: Contains OK/NOK SQL examples and documentation
- **LangGraph**: Orchestrates the multi-agent workflow

## Coding Guidelines
- Follow Python best practices and type hints
- Use LangChain Expression Language (LCEL) for agent chains
- Implement proper error handling for database connections
- Use async/await patterns where appropriate
- Follow the established project structure in `/src`

## Standards
- SQL examples marked as "OK" represent correct standards
- SQL examples marked as "NOK" represent violations that need correction
- Each NOK example has a corresponding OK version for reference
- Focus on Teradata-specific SQL optimizations and best practices
