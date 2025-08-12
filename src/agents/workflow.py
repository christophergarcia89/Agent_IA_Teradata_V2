"""
LangGraph Workflow that orchestrates the three agents for SQL query analysis and optimization.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import Annotated, TypedDict

from src.agents.sql_reviewer import SQLReviewerAgent, SQLReviewResult
from src.agents.explain_generator import EnhancedExplainGenerator, EnhancedExplainResult
from src.agents.explain_interpreter import ExplainInterpreterAgent, ExplainAnalysis
from src.rag.vector_store import VectorStore
from config.settings import settings


class WorkflowState(TypedDict):
    """State maintained throughout the workflow."""
    # Input
    original_query: str
    
    # Agent outputs
    review_result: Optional[SQLReviewResult]
    explain_result: Optional[EnhancedExplainResult]
    analysis_result: Optional[ExplainAnalysis]
    
    # Additional data
    corrected_query: str
    final_recommendations: List[str]
    workflow_status: str
    error_messages: List[str]
    
    # Messages for conversation
    messages: Annotated[List[AnyMessage], add_messages]


@dataclass
class WorkflowResult:
    """Final result of the complete workflow."""
    original_query: str
    corrected_query: str
    is_standards_compliant: bool
    standards_violations: List[str]
    explain_plan: str
    performance_assessment: str
    optimization_suggestions: List[Dict[str, Any]]
    final_recommendations: List[str]
    success: bool
    error_messages: List[str]


class TeradataWorkflow:
    """Main workflow that orchestrates SQL analysis using LangGraph."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize shared vector store
        self.vector_store = VectorStore()
        
        # Initialize agents with shared vector store
        self.sql_reviewer = SQLReviewerAgent(vector_store=self.vector_store)
        self.explain_generator = EnhancedExplainGenerator()
        self.explain_interpreter = ExplainInterpreterAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    async def initialize(self) -> None:
        """Initialize all agents and ensure ChromaDB is fully loaded."""
        try:
            self.logger.info("Initializing Teradata Workflow...")
            
            # Step 1: Initialize shared vector store and WAIT for complete loading
            self.logger.info("[STEP 1] Initializing and loading ChromaDB...")
            await self._initialize_vector_store_completely()
            
            # Step 2: Initialize other agents in parallel (they depend on vector store)
            self.logger.info("[STEP 2] Initializing agents...")
            await asyncio.gather(
                self.sql_reviewer.initialize(),
                self.explain_generator.initialize(),
                # explain_interpreter doesn't need async initialization
            )
            
            # Step 3: Validate that everything is ready
            self.logger.info("[STEP 3] Validating system readiness...")
            await self._validate_system_readiness()
            self.logger.info("[OK] Step 3: Validating system readiness...")
            await self._validate_system_readiness()
            
            self.logger.info("🎉 Teradata Workflow initialized successfully and ready for queries")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow: {e}")
            raise
    
    async def _initialize_vector_store_completely(self) -> None:
        """Initialize vector store and wait for complete document loading."""
        try:
            # Initialize ChromaDB structure
            await self.vector_store.initialize()
            
            # Wait for documents to be completely loaded
            self.logger.info("⏳ Waiting for document loading to complete...")
            
            # Check if documents are already loaded
            doc_count = await self.vector_store._get_collection_count()
            
            if doc_count > 0:
                self.logger.info(f"[DOCS] Found {doc_count} documents already loaded")
                return
            
            # If no documents, trigger loading and wait
            self.logger.info("📥 No documents found, triggering complete loading...")
            
            # Load knowledge base completely (not in background)
            import asyncio
            await asyncio.wait_for(
                self.vector_store.load_knowledge_base(),
                timeout=300.0  # 5 minutes timeout for complete loading
            )
            
            # Verify documents were loaded
            final_count = await self.vector_store._get_collection_count()
            self.logger.info(f"[OK] ChromaDB fully loaded with {final_count} documents")
            
            if final_count == 0:
                self.logger.warning("⚠️ Warning: ChromaDB initialized but no documents loaded")
            
        except asyncio.TimeoutError:
            self.logger.error("[ERROR] Timeout loading documents - system may have reduced functionality")
            raise
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading ChromaDB completely: {e}")
            raise
    
    async def _validate_system_readiness(self) -> None:
        """Validate that all system components are ready."""
        try:
            readiness_checks = []
            
            # Check 1: Vector store has documents
            doc_count = await self.vector_store._get_collection_count()
            readiness_checks.append(("ChromaDB documents", doc_count > 0, f"{doc_count} docs"))
            
            # Check 2: SQL reviewer can get context
            test_examples = await self.vector_store.search_similar_examples("SELECT * FROM test", k=1)
            readiness_checks.append(("SQL examples search", len(test_examples) > 0, f"{len(test_examples)} found"))
            
            # Check 3: Documentation search
            test_docs = await self.vector_store.search_documentation("SQL standards", k=1)
            readiness_checks.append(("Documentation search", len(test_docs) > 0, f"{len(test_docs)} found"))
            
            # Log readiness results
            all_ready = True
            for check_name, is_ready, details in readiness_checks:
                status = "[OK]" if is_ready else "[ERROR]"
                self.logger.info(f"  {status} {check_name}: {details}")
                all_ready = all_ready and is_ready
            
            if not all_ready:
                raise RuntimeError("System readiness validation failed - some components not ready")
                
            self.logger.info("🎯 All system components validated and ready")
            
        except Exception as e:
            self.logger.error(f"System readiness validation failed: {e}")
            raise
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with robust error handling."""
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (agents)
        workflow.add_node("sql_review", self._sql_review_node)
        workflow.add_node("explain_generation", self._explain_generation_node)
        workflow.add_node("explain_interpretation", self._explain_interpretation_node)
        workflow.add_node("finalization", self._finalization_node)
        
        # Define the workflow edges with robust conditional routing
        workflow.set_entry_point("sql_review")
        
        # Conditional routing after SQL review
        workflow.add_conditional_edges(
            "sql_review",
            self._should_continue_after_review,
            {
                "continue": "explain_generation",
                "stop": "finalization"
            }
        )
        
        # Conditional routing after EXPLAIN generation
        workflow.add_conditional_edges(
            "explain_generation",
            self._should_continue_after_explain,
            {
                "continue": "explain_interpretation",
                "skip": "finalization"
            }
        )
        
        workflow.add_edge("explain_interpretation", "finalization")
        workflow.add_edge("finalization", END)
        
        return workflow.compile()
    
    async def _sql_review_node(self, state: WorkflowState) -> WorkflowState:
        """Node for SQL review using RAG with robust error handling."""
        try:
            self.logger.info("[REVIEW] Executing SQL review node")
            
            # Validate input
            if not state["original_query"].strip():
                raise ValueError("Empty or invalid SQL query provided")
            
            # Review the original query
            review_result = await self.sql_reviewer.review_query(state["original_query"])
            
            # Validate review result
            if not review_result:
                raise ValueError("SQL reviewer returned empty result")
            
            # Update state with successful review
            state["review_result"] = review_result
            state["corrected_query"] = review_result.corrected_query
            state["workflow_status"] = "sql_review_completed"
            
            # Add informative message
            violations_msg = f"Found {len(review_result.violations)} violations" if review_result.violations else "No violations found"
            state["messages"].append({
                "type": "ai",
                "content": f"[OK] SQL Review completed. Compliant: {review_result.is_compliant}. {violations_msg}."
            })
            
            self.logger.info(f"[OK] SQL review completed - Compliant: {review_result.is_compliant}, Violations: {len(review_result.violations)}")
            return state
            
        except Exception as e:
            error_msg = f"SQL review node error: {str(e)}"
            self.logger.error(error_msg)
            
            # Don't fail catastrophically - update state with error info and continue
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "sql_review_error"
            
            # Create minimal review result to allow workflow continuation
            state["review_result"] = SQLReviewResult(
                is_compliant=False,
                violations=[f"Review error: {str(e)}"],
                corrected_query=state["original_query"],  # Use original as fallback
                recommendations=["Manual review recommended due to automated review failure"],
                confidence_score=0.0,
                used_examples=[]
            )
            state["corrected_query"] = state["original_query"]
            
            state["messages"].append({
                "type": "ai", 
                "content": f"⚠️ SQL Review encountered an error but workflow will continue: {str(e)[:100]}..."
            })
            
            return state
    
    async def _explain_generation_node(self, state: WorkflowState) -> WorkflowState:
        """Node for generating EXPLAIN plan with timeout and circuit breaker protection."""
        try:
            self.logger.info("[EXPLAIN] Executing EXPLAIN generation node")
            
            # Use corrected query if available and valid, otherwise original
            query_to_explain = state["corrected_query"] if state["corrected_query"].strip() else state["original_query"]
            
            if not query_to_explain.strip():
                raise ValueError("No valid query available for EXPLAIN generation")
            
            self.logger.info(f"Generating EXPLAIN for query: {query_to_explain[:100]}...")
            
            # Generate EXPLAIN plan with timeout protection
            explain_result = await self.explain_generator.generate_explain(query_to_explain)
            
            # Update state with result (successful or failed)
            state["explain_result"] = explain_result
            
            if explain_result.success:
                state["workflow_status"] = "explain_generation_completed"
                status_msg = f"[OK] EXPLAIN plan generated successfully"
                cost_info = f"Cost: {explain_result.query_cost}" if explain_result.query_cost else "Cost: Unknown"
                
                state["messages"].append({
                    "type": "ai",
                    "content": f"{status_msg}. {cost_info}"
                })
                
                self.logger.info(f"[OK] EXPLAIN generation successful - {cost_info}")
                
            else:
                # EXPLAIN failed but don't stop the workflow - use fallback mode
                state["workflow_status"] = "explain_generation_fallback"
                error_info = explain_result.error_message or "Unknown error"
                
                state["messages"].append({
                    "type": "ai",
                    "content": f"⚠️ EXPLAIN plan generation failed: {error_info}. Workflow will continue with available data."
                })
                
                self.logger.warning(f"⚠️ EXPLAIN generation failed: {error_info}")
            
            return state
            
        except Exception as e:
            error_msg = f"EXPLAIN generation node error: {str(e)}"
            self.logger.error(error_msg)
            
            # Don't fail catastrophically - create fallback result
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "explain_generation_error"
            
            # Create fallback EXPLAIN result
            from src.agents.explain_generator import EnhancedExplainResult
            state["explain_result"] = EnhancedExplainResult(
                original_query=query_to_explain if 'query_to_explain' in locals() else state["original_query"],
                explain_plan="",
                query_cost=None,
                execution_time=0.0,
                success=False,
                error_message=f"EXPLAIN generation failed: {str(e)}",
                fallback_used=True,
                circuit_breaker_active=False
            )
            
            state["messages"].append({
                "type": "ai",
                "content": f"[ERROR] EXPLAIN generation failed: {str(e)[:100]}... Workflow will continue without EXPLAIN plan."
            })
            
            return state
    
    async def _explain_interpretation_node(self, state: WorkflowState) -> WorkflowState:
        """Node for interpreting EXPLAIN plan and providing suggestions with robust error handling."""
        try:
            self.logger.info("🎯 Executing EXPLAIN interpretation node")
            
            # Validate we have required data
            if not state["explain_result"]:
                raise ValueError("No EXPLAIN result available for interpretation")
                
            if not state["explain_result"].success:
                raise ValueError(f"EXPLAIN generation failed: {state['explain_result'].error_message}")
            
            if not state["explain_result"].explain_plan:
                raise ValueError("EXPLAIN plan content is empty")
            
            # Get query for analysis
            query_for_analysis = state["corrected_query"] if state["corrected_query"].strip() else state["original_query"]
            
            self.logger.info(f"Interpreting EXPLAIN plan for query: {query_for_analysis[:100]}...")
            
            # Analyze the EXPLAIN plan
            analysis_result = await self.explain_interpreter.analyze_explain_plan(
                query_for_analysis, 
                state["explain_result"]
            )
            
            # Validate analysis result
            if not analysis_result:
                raise ValueError("EXPLAIN interpreter returned empty result")
            
            # Update state with successful analysis
            state["analysis_result"] = analysis_result
            state["workflow_status"] = "explain_interpretation_completed"
            
            # Add informative message
            suggestions_count = len(analysis_result.suggestions) if analysis_result.suggestions else 0
            performance_info = analysis_result.overall_performance or "Unknown"
            
            state["messages"].append({
                "type": "ai",
                "content": f"[OK] EXPLAIN analysis completed. Performance: {performance_info}. "
                          f"Generated {suggestions_count} optimization suggestions."
            })
            
            self.logger.info(f"[OK] EXPLAIN interpretation successful - Performance: {performance_info}, Suggestions: {suggestions_count}")
            return state
            
        except Exception as e:
            error_msg = f"EXPLAIN interpretation node error: {str(e)}"
            self.logger.error(error_msg)
            
            # Don't fail catastrophically - create fallback result
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "explain_interpretation_error"
            
            # Create fallback analysis result
            from src.agents.explain_interpreter import ExplainAnalysis, OptimizationSuggestion
            state["analysis_result"] = ExplainAnalysis(
                query=query_for_analysis if 'query_for_analysis' in locals() else state["original_query"],
                explain_plan=state["explain_result"].explain_plan if state.get("explain_result") else "",
                overall_performance="Error - Unable to analyze",
                performance_score=0.0,
                bottlenecks=["Analysis failed due to error"],
                suggestions=[
                    OptimizationSuggestion(
                        issue="Analysis Error",
                        suggestion=f"Manual analysis recommended: {str(e)}",
                        priority="high",
                        impact="unknown",
                        implementation="Manual review required"
                    )
                ],
                confidence_level="low",
                analysis_timestamp=None
            )
            
            state["messages"].append({
                "type": "ai",
                "content": f"⚠️ EXPLAIN interpretation failed: {str(e)[:100]}... Workflow will continue with basic recommendations."
            })
            
            return state
            
            # Add message
            state["messages"].append({
                "type": "ai",
                "content": f"EXPLAIN analysis completed. Performance: {analysis_result.overall_performance}. "
                          f"Found {len(analysis_result.suggestions)} optimization suggestions."
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Error in explain interpretation node: {str(e)}"
            self.logger.error(error_msg)
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "error"
            return state
    
    async def _finalization_node(self, state: WorkflowState) -> WorkflowState:
        """Final node to compile results and recommendations."""
        try:
            self.logger.info("Executing finalization node")
            
            # Compile final recommendations
            final_recommendations = []
            
            # Add standards compliance recommendations
            if state["review_result"]:
                if not state["review_result"].is_compliant:
                    final_recommendations.extend([
                        f"Standards violation: {violation}"
                        for violation in state["review_result"].violations
                    ])
                final_recommendations.extend(state["review_result"].recommendations)
            
            # Add performance optimization recommendations
            if state["analysis_result"]:
                for suggestion in state["analysis_result"].suggestions:
                    final_recommendations.append(
                        f"[{suggestion.priority.upper()}] {suggestion.issue}: {suggestion.suggestion}"
                    )
            
            # Update state
            state["final_recommendations"] = final_recommendations
            state["workflow_status"] = "completed"
            
            # Add final message
            state["messages"].append({
                "type": "ai",
                "content": f"Workflow completed successfully. Generated {len(final_recommendations)} recommendations."
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Error in finalization node: {str(e)}"
            self.logger.error(error_msg)
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "error"
            return state
    
    def _should_continue_after_review(self, state: WorkflowState) -> str:
        """Decide whether to continue with EXPLAIN generation after SQL review."""
        # Stop only if there's a critical system error (not just review errors)
        if state["workflow_status"] == "error":
            self.logger.info("[ERROR] Stopping workflow due to critical system error in SQL review")
            return "stop"
        
        # Check if we have a review result (even if it has errors)
        if not state["review_result"]:
            self.logger.warning("⚠️ No review result available, stopping workflow")
            return "stop"
        
        # Continue regardless of violations - EXPLAIN can still provide valuable insights
        violations_count = len(state["review_result"].violations) if state["review_result"].violations else 0
        
        if violations_count > 0:
            self.logger.info(f"📋 Found {violations_count} violations, but continuing with EXPLAIN generation for complete analysis")
        else:
            self.logger.info("[OK] No violations found, proceeding with EXPLAIN generation")
        
        # Even if there were review errors, continue if we have a query to analyze
        if state["workflow_status"] in ["sql_review_completed", "sql_review_error"]:
            query_to_use = state["corrected_query"] if state["corrected_query"].strip() else state["original_query"]
            if query_to_use.strip():
                self.logger.info("[EXPLAIN] Proceeding to EXPLAIN generation")
                return "continue"
        
        self.logger.warning("⚠️ No valid query available for EXPLAIN generation")
        return "stop"
    
    def _should_continue_after_explain(self, state: WorkflowState) -> str:
        """Decide whether to continue with EXPLAIN interpretation after generation."""
        # Stop if there's a critical error
        if state["workflow_status"] == "error":
            self.logger.info("Stopping workflow due to error in EXPLAIN generation")
            return "skip"
        
        # Check if we have an EXPLAIN result
        if not state["explain_result"]:
            self.logger.warning("No EXPLAIN result available, skipping interpretation")
            return "skip"
        
        # Check if EXPLAIN was successful
        if not state["explain_result"].success:
            self.logger.info("EXPLAIN generation failed, but continuing to finalization with available data")
            return "skip"
        
        # Check if we have actual plan content
        if not state["explain_result"].explain_plan:
            self.logger.info("No EXPLAIN plan content available, skipping interpretation")
            return "skip"
        
        self.logger.info("EXPLAIN plan available, proceeding with interpretation")
        return "continue"
    
    async def process_query(self, sql_query: str) -> WorkflowResult:
        """Process a SQL query through the complete workflow."""
        try:
            self.logger.info("Starting query processing workflow")
            
            # Initialize state
            initial_state = WorkflowState(
                original_query=sql_query,
                review_result=None,
                explain_result=None,
                analysis_result=None,
                corrected_query="",
                final_recommendations=[],
                workflow_status="started",
                error_messages=[],
                messages=[]
            )
            
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Create result object
            result = self._create_workflow_result(final_state)
            
            self.logger.info(f"Workflow completed with status: {final_state['workflow_status']}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg)
            
            # Return error result
            return WorkflowResult(
                original_query=sql_query,
                corrected_query="",
                is_standards_compliant=False,
                standards_violations=[],
                explain_plan="",
                performance_assessment="Error",
                optimization_suggestions=[],
                final_recommendations=[],
                success=False,
                error_messages=[error_msg]
            )
    
    def _create_workflow_result(self, state: WorkflowState) -> WorkflowResult:
        """Create the final workflow result from state."""
        success = state["workflow_status"] == "completed" and not state["error_messages"]
        
        # Extract data from agent results
        is_compliant = state["review_result"].is_compliant if state["review_result"] else False
        violations = state["review_result"].violations if state["review_result"] else []
        explain_plan = state["explain_result"].explain_plan if state["explain_result"] else ""
        performance = state["analysis_result"].overall_performance if state["analysis_result"] else "Unknown"
        
        # Format optimization suggestions
        optimization_suggestions = []
        if state["analysis_result"]:
            for suggestion in state["analysis_result"].suggestions:
                optimization_suggestions.append({
                    "issue": suggestion.issue,
                    "suggestion": suggestion.suggestion,
                    "priority": suggestion.priority,
                    "impact": suggestion.impact,
                    "implementation": suggestion.implementation
                })
        
        return WorkflowResult(
            original_query=state["original_query"],
            corrected_query=state["corrected_query"],
            is_standards_compliant=is_compliant,
            standards_violations=violations,
            explain_plan=explain_plan,
            performance_assessment=performance,
            optimization_suggestions=optimization_suggestions,
            final_recommendations=state["final_recommendations"],
            success=success,
            error_messages=state["error_messages"]
        )
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.explain_generator.cleanup()
            self.logger.info("Workflow cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get the current status of the workflow."""
        return {
            "agents_initialized": True,
            "workflow_ready": True,
            "supported_operations": [
                "SQL standards review",
                "EXPLAIN plan generation",
                "Performance analysis",
                "Optimization suggestions"
            ]
        }
