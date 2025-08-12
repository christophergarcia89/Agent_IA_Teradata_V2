"""
SQL Reviewer de línea de comandos
Herramienta simple para analizar consultas SQL desde la consola
"""

import asyncio
import os
import sys
import argparse
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def analyze_sql_query(query: str, use_full_rag: bool = True):
    """Analizar una consulta SQL específica."""
    
    print("[TEST] SQL REVIEWER - TERADATA")
    print("=" * 40)
    print(f"[TIME] {datetime.now().strftime('%H:%M:%S')}")
    print(f"[RAG] Modo: {'COMPLETO con ChromaDB' if use_full_rag else 'SIMPLIFICADO con mock'}")
    print(f"[QUERY] Consulta:")
    print("-" * 20)
    print(query.strip())
    print("-" * 20)
    
    try:
        # Importar componentes necesarios
        from src.agents.sql_reviewer import SQLReviewerAgent
        from langchain.schema import Document
        
        # Crear reviewer
        reviewer = SQLReviewerAgent()
        print("[OK] SQL Reviewer inicializado")
        
        if use_full_rag:
            print("[LOADING] Inicializando RAG completo con ChromaDB...")
            print("[INFO] Primera carga puede tomar 3-5 minutos...")
            
            # Inicializar con RAG completo
            await reviewer.initialize()
            print("[OK] RAG completo cargado exitosamente")
            print(f"[RAG] Vector Store con {reviewer.vector_store.get_collection_count()} documentos")
            
            # Usar el método completo del reviewer
            print("[PROCESSING] Analizando con contexto RAG completo...")
            result = await reviewer.review_query(query)
            
        else:
            print("[MOCK] Usando contexto simplificado...")
            # Mantener el código mock original para comparación
            mock_examples = [
                Document(
                    page_content="""-- EJEMPLO OK: Método recomendado
DROP TABLE TMP.TABLA_TEMP;
CREATE MULTISET TABLE TMP.TABLA_TEMP AS (
    SELECT rut, nombre, estado
    FROM PROD.CLIENTES
    WHERE activo = 1
) WITH DATA;""",
                    metadata={"example_type": "OK", "category": "GENERAL"}
                ),
                Document(
                    page_content="""-- EJEMPLO NOK: Problemas comunes
UPDATE EDW.CLIENTES SET estado = 'ACTIVO'
select * from bcimkt.tabla1 where campo<>1;""",
                    metadata={"example_type": "NOK", "category": "GENERAL"}
                )
            ]
            
            mock_docs = [
                Document(
                    page_content="""REGLAS CRÍTICAS TERADATA:
1. PROHIBIDO UPDATE directo - usar CREATE temporal + DELETE + INSERT
2. PROHIBIDO esquemas: EDW, BCIMKT, MKT_*
3. Keywords en MAYÚSCULAS
4. No usar asterisco (*) en SELECT
5. Alias de tabla: exactamente 3 caracteres
6. Comas a la izquierda
7. Reemplazar <> por NOT(campo1=campo2)""",
                    metadata={"filename": "normativa.txt"}
                )
            ]
            
            # Generar prompt con datos mock
            formatted_prompt = reviewer.review_prompt.format_messages(
                original_query=query,
                similar_examples=reviewer._format_examples(mock_examples),
                documentation=reviewer._format_documentation(mock_docs),
                ok_examples=reviewer._format_examples(mock_examples),
                format_instructions=reviewer.output_parser.get_format_instructions()
            )
            
            print("[WAIT] Enviando a Azure OpenAI...")
            
            # Analizar con Azure OpenAI
            response = await reviewer.llm.ainvoke(formatted_prompt)
            result = reviewer.output_parser.parse(response.content)
        
        # Mostrar resultados
        print("\n[RESULTS] RESULTADOS DEL ANÁLISIS")
        print("=" * 30)
        
        # Estado general
        status_icon = "[OK]" if result.is_compliant else "[ERROR]"
        status_text = "CUMPLE ESTÁNDARES" if result.is_compliant else "NO CUMPLE ESTÁNDARES"
        print(f"{status_icon} Estado: {status_text}")
        print(f"[CONFIDENCE] Confianza: {result.confidence_score:.1%}")
        
        # Violaciones
        if result.violations:
            print(f"\n[VIOLATIONS] VIOLACIONES ENCONTRADAS ({len(result.violations)}):")
            for i, violation in enumerate(result.violations, 1):
                print(f"   {i}. {violation}")
        else:
            print(f"\n[OK] No se encontraron violaciones")
        
        # Recomendaciones
        if result.recommendations:
            print(f"\n[RECOMMENDATIONS] RECOMENDACIONES:")
            for i, rec in enumerate(result.recommendations[:3], 1):  # Máximo 3
                print(f"   {i}. {rec}")
        
        # Consulta corregida
        if result.corrected_query.strip() != query.strip():
            print(f"\n[FIXED] CONSULTA CORREGIDA:")
            print("-" * 30)
            corrected_lines = result.corrected_query.strip().split('\n')
            for i, line in enumerate(corrected_lines, 1):
                if line.strip():  # Solo líneas no vacías
                    print(f"{i:2d}. {line}")
            print("-" * 30)
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error durante el análisis: {str(e)}")
        return False

def parse_multiline_query(query_arg: str) -> str:
    """Procesar consulta que puede venir con saltos de línea escapados."""
    # Reemplazar \n literales por saltos de línea reales
    query = query_arg.replace('\\n', '\n')
    # Reemplazar múltiples espacios por espacios simples
    import re
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

async def interactive_mode(use_rag: bool = True):
    """Modo interactivo para ingresar múltiples consultas."""
    rag_mode = "RAG COMPLETO" if use_rag else "MOCK"
    print(f"[INTERACTIVE] MODO INTERACTIVO SQL REVIEWER ({rag_mode})")
    print("=" * 35)
    print("[TIPS] Consejos:")
    print("   • Escribe 'quit' para salir")
    print("   • Escribe 'help' para ver ejemplos")
    print("   • Puedes pegar consultas de múltiples líneas")
    
    while True:
        print("\n" + "-" * 40)
        
        # Leer consulta del usuario
        print("[INPUT] Ingresa tu consulta SQL:")
        lines = []
        
        while True:
            try:
                line = input("> " if not lines else "  ")
                
                if line.strip().lower() == 'quit':
                    print("[BYE] ¡Hasta luego!")
                    return
                
                if line.strip().lower() == 'help':
                    print("\n[EXAMPLES] EJEMPLOS DE CONSULTAS:")
                    print("1. UPDATE EDW.CLIENTES SET estado = 'ACTIVO'")
                    print("2. select * from bcimkt.tabla1 where id<>1")
                    print("3. SELECT rut, nombre FROM PROD.CLIENTES")
                    print("4. CREATE TABLE EDW.NUEVA AS SELECT * FROM OTRA")
                    break
                
                lines.append(line)
                
                # Si la línea termina con ; o está vacía, procesar
                if line.strip().endswith(';') or (line.strip() == '' and lines):
                    break
                    
            except KeyboardInterrupt:
                print("\n[BYE] ¡Hasta luego!")
                return
            except EOFError:
                break
        
        if lines and lines != ['help']:
            query = '\n'.join(lines).strip()
            if query:
                await analyze_sql_query(query, use_rag)

async def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='SQL Reviewer para analizar consultas de Teradata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s -q "UPDATE EDW.CLIENTES SET estado = 'ACTIVO'"
  %(prog)s -f mi_consulta.sql
  %(prog)s -i
  %(prog)s --query "SELECT * FROM tabla WHERE id<>1"
        """
    )
    
    parser.add_argument('-q', '--query', type=str, 
                       help='Consulta SQL a analizar (entre comillas)')
    parser.add_argument('-f', '--file', type=str, 
                       help='Archivo .sql con la consulta a analizar')
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Modo interactivo para múltiples consultas')
    parser.add_argument('--mock', action='store_true',
                       help='Usar contexto mock simplificado (más rápido)')
    parser.add_argument('--full-rag', action='store_true',
                       help='Usar RAG completo con ChromaDB (más preciso, más lento)')
    parser.add_argument('--version', action='version', version='SQL Reviewer 1.0')
    
    args = parser.parse_args()
    
    # Si no hay argumentos, mostrar ayuda
    if not any([args.query, args.file, args.interactive]):
        parser.print_help()
        return
    
    # Determinar modo RAG
    if args.mock:
        use_rag = False
        print("[MODE] Usando contexto MOCK (rápido)")
    elif args.full_rag:
        use_rag = True
        print("[MODE] Usando RAG COMPLETO (lento, primera vez)")
    else:
        # Por defecto usar RAG completo
        use_rag = True
        print("[MODE] Usando RAG COMPLETO por defecto")
    
    print("[TEST] SQL REVIEWER - TERADATA")
    print("=" * 30)
    
    if args.interactive:
        await interactive_mode(use_rag)
    elif args.query:
        query = parse_multiline_query(args.query)
        await analyze_sql_query(query, use_rag)
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                query = f.read().strip()
            print(f"[FILE] Leyendo consulta del archivo: {args.file}")
            await analyze_sql_query(query, use_rag)
        except FileNotFoundError:
            print(f"[ERROR] Archivo no encontrado: {args.file}")
        except Exception as e:
            print(f"[ERROR] Error leyendo archivo: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
