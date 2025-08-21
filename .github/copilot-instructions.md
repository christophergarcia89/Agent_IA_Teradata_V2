# Instrucciones de Copilot

<!-- Usa este archivo para proporcionar instrucciones personalizadas específicas del espacio de trabajo a Copilot. Para más detalles, visita https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Contexto del Proyecto
Este es un proyecto LangGraph LCEL para revisión y optimización de consultas SQL de Teradata usando RAG (Generación Aumentada por Recuperación).

## Componentes Clave
- **Sistema RAG**: Utiliza base de conocimiento interna con estándares SQL y ejemplos
- **Tres Agentes**: 
  1. Revisor de Consultas SQL (usa RAG para cumplimiento de estándares)
  2. Generador de Explicaciones de Consultas (usa teradata-mcp-server)
  3. Intérprete de Explicaciones (proporciona sugerencias de optimización)
- **Base de Conocimiento**: Contiene ejemplos SQL OK/NOK y documentación
- **LangGraph**: Orquesta el flujo de trabajo multi-agente

## Directrices de Codificación
- Seguir las mejores prácticas de Python y anotaciones de tipo
- Usar LangChain Expression Language (LCEL) para cadenas de agentes
- Implementar manejo adecuado de errores para conexiones de base de datos
- Usar patrones async/await donde sea apropiado
- Seguir la estructura de proyecto establecida en `/src`

## Estándares
- Los ejemplos SQL marcados como "OK" representan estándares correctos
- Los ejemplos SQL marcados como "NOK" representan violaciones que necesitan corrección
- Cada ejemplo NOK tiene una versión OK correspondiente para referencia
- Enfocarse en optimizaciones SQL específicas de Teradata y mejores prácticas
