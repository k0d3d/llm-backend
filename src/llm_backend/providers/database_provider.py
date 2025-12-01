"""
Database provider implementation for client-side DuckDB operations

Follows the AIProvider interface pattern with Pydantic AI agents for:
- Schema inspection (read-only, no approval)
- Database queries (SELECT only, auto-execute)
- SQL execution (INSERT/UPDATE/DELETE, requires HITL approval)
"""

import time
from typing import Dict, Any, List, Optional
from enum import Enum

from llm_backend.core.providers.base import (
    AIProvider, ProviderPayload, ProviderResponse,
    ProviderCapabilities, ValidationIssue, OperationType
)


class DatabaseOperationType(str, Enum):
    """Database-specific operation types"""
    SCHEMA_INSPECTION = "schema_inspection"
    QUERY = "query"
    EXECUTION = "execution"


class DatabasePayload(ProviderPayload):
    """Database-specific payload"""
    operation: DatabaseOperationType
    sql_query: Optional[str] = None
    sql_statements: Optional[List[str]] = None
    description: str = ""
    current_schema: Dict[str, Any] = {}
    document_data: Optional[Dict[str, Any]] = None
    requires_approval: bool = False
    auto_execute: bool = False


class DatabaseProvider(AIProvider):
    """
    Provider implementation for client-side DuckDB database operations

    Uses Pydantic AI agents for type-safe database operations:
    - SchemaInspectionAgent: Lists tables/columns (no approval needed)
    - DatabaseQueryAgent: Creates SELECT queries (auto-execute)
    - SQLExecutionAgent: Creates INSERT/UPDATE/DELETE (requires HITL approval)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DatabaseProvider

        Args:
            config: Optional configuration with current_schema, document_url, etc.
        """
        super().__init__(config or {})
        self.current_schema = config.get("current_schema", {"tables": []}) if config else {"tables": []}
        self.document_url = config.get("document_url")
        self._orchestrator = None

    def get_capabilities(self) -> ProviderCapabilities:
        """Return database provider capabilities"""
        return ProviderCapabilities(
            name="DuckDB Client-Side Database",
            description="Client-side database operations with HITL approval for data modifications",
            version="1.0.0",
            input_schema={
                "prompt": "Natural language request for database operation",
                "current_schema": "Current database schema from frontend",
                "document_data": "Extracted data from PDFs for INSERT operations"
            },
            supported_operations=[
                OperationType.DATABASE_QUERY,
                OperationType.DATABASE_WRITE,
                OperationType.DATABASE_SCHEMA
            ],
            safety_features=["hitl_approval_for_writes", "select_only_auto_execute", "sql_injection_prevention"],
            rate_limits={"requests_per_minute": 1000},  # Client-side, no API limits
            max_input_size=100 * 1024 * 1024,  # 100MB for DuckDB
            cost_per_request=0.0  # Free, runs in browser
        )

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator reference to access state"""
        self._orchestrator = orchestrator
        print(f"📋 Orchestrator linked to DatabaseProvider")

    def _detect_operation_type(self, prompt: str) -> DatabaseOperationType:
        """
        Detect what type of database operation the user wants

        Args:
            prompt: User's natural language request

        Returns:
            DatabaseOperationType (schema_inspection, query, or execution)
        """
        prompt_lower = prompt.lower().strip()

        # Schema inspection keywords
        schema_keywords = [
            "show tables", "list tables", "describe table", "table structure",
            "show schema", "what tables", "database structure", "show columns"
        ]
        if any(keyword in prompt_lower for keyword in schema_keywords):
            return DatabaseOperationType.SCHEMA_INSPECTION

        # Query keywords (SELECT)
        query_keywords = [
            "select", "show me", "get", "find", "search", "query",
            "list all", "count", "sum", "average", "group by"
        ]
        if any(keyword in prompt_lower for keyword in query_keywords):
            # Make sure it's not an INSERT disguised as "show me how to insert"
            if not any(write_keyword in prompt_lower for write_keyword in ["insert", "create", "update", "delete", "drop"]):
                return DatabaseOperationType.QUERY

        # Execution keywords (INSERT, UPDATE, DELETE, CREATE)
        execution_keywords = [
            "insert", "add", "create table", "update", "modify",
            "delete", "remove", "drop", "alter table"
        ]
        if any(keyword in prompt_lower for keyword in execution_keywords):
            return DatabaseOperationType.EXECUTION

        # Default to query for safety (read-only)
        return DatabaseOperationType.QUERY

    async def create_payload(
        self,
        prompt: str,
        attachments: List[str],
        operation_type: OperationType,
        config: Dict,
        conversation: Optional[List[Dict[str, str]]] = None,
        hitl_edits: Dict = None
    ) -> DatabasePayload:
        """
        Create database-specific payload using Pydantic AI agents

        Args:
            prompt: User's natural language request
            attachments: Attached documents (PDFs for data extraction)
            operation_type: Generic operation type
            config: Configuration with current_schema, document_data, etc.
            conversation: Chat history
            hitl_edits: Human edits to apply

        Returns:
            DatabasePayload with SQL and metadata
        """
        from llm_backend.agents.database_schema_agent import inspect_schema
        from llm_backend.agents.database_query_agent import create_database_query
        from llm_backend.agents.database_execution_agent import create_sql_execution

        print("🗄️ Creating database payload with Pydantic AI agents")
        print(f"📝 User prompt: '{prompt[:100]}...'")

        # Get current schema and document data from config
        current_schema = config.get("current_schema", self.current_schema)
        document_data = config.get("document_data")

        # Detect operation type
        db_operation = self._detect_operation_type(prompt)
        print(f"🔍 Detected operation type: {db_operation.value}")

        # Route to appropriate Pydantic AI agent
        if db_operation == DatabaseOperationType.SCHEMA_INSPECTION:
            print("📊 Using Schema Inspection Agent")
            result = await inspect_schema(
                query=prompt,
                current_schema=current_schema
            )

            return DatabasePayload(
                operation=db_operation,
                sql_query=None,  # No SQL for schema inspection
                description=result.message,
                current_schema=current_schema,
                requires_approval=False,
                auto_execute=True,  # Auto-execute schema inspection
                metadata={
                    "agent_output": result.model_dump(),
                    "agent_id": result.agentId,
                    "table_count": result.tableCount
                }
            )

        elif db_operation == DatabaseOperationType.QUERY:
            print("🔎 Using Database Query Agent")
            result = await create_database_query(
                user_request=prompt,
                current_schema=current_schema
            )

            # Check if query creation failed
            if result.status == "error" or not result.validation.is_valid:
                return DatabasePayload(
                    operation=db_operation,
                    sql_query="",
                    description=result.description or "Query creation failed",
                    current_schema=current_schema,
                    requires_approval=False,
                    auto_execute=False,
                    metadata={
                        "agent_output": result.model_dump(),
                        "error": result.validation.error
                    }
                )

            return DatabasePayload(
                operation=db_operation,
                sql_query=result.sql_query,
                description=result.description,
                current_schema=current_schema,
                requires_approval=False,
                auto_execute=True,  # Auto-execute SELECT queries
                metadata={
                    "agent_output": result.model_dump(),
                    "agent_id": result.agentId,
                    "validation": result.validation.model_dump()
                }
            )

        else:  # EXECUTION
            print("⚙️ Using SQL Execution Agent")

            # Determine SQL operation type from prompt
            prompt_lower = prompt.lower()
            if "insert" in prompt_lower or "add" in prompt_lower:
                sql_operation = "INSERT"
            elif "create table" in prompt_lower or "create" in prompt_lower:
                sql_operation = "CREATE"
            elif "update" in prompt_lower or "modify" in prompt_lower:
                sql_operation = "UPDATE"
            elif "delete" in prompt_lower or "remove" in prompt_lower:
                sql_operation = "DELETE"
            elif "alter" in prompt_lower:
                sql_operation = "ALTER"
            else:
                sql_operation = "INSERT"  # Default

            result = await create_sql_execution(
                user_request=prompt,
                operation_type=sql_operation,
                current_schema=current_schema,
                document_data=document_data
            )

            # Check if execution creation failed
            if result.status == "error" or not result.validation.is_valid:
                return DatabasePayload(
                    operation=db_operation,
                    sql_statements=[],
                    description=result.description or "SQL creation failed",
                    current_schema=current_schema,
                    document_data=document_data,
                    requires_approval=True,  # Still require approval even for errors
                    auto_execute=False,
                    metadata={
                        "agent_output": result.model_dump(),
                        "error": result.validation.issues,
                        "blocking_issues": result.validation.blocking_issues
                    }
                )

            return DatabasePayload(
                operation=db_operation,
                sql_statements=result.sql_statements,
                description=result.description,
                current_schema=current_schema,
                document_data=document_data,
                requires_approval=True,  # ALWAYS require approval for writes
                auto_execute=False,
                metadata={
                    "agent_output": result.model_dump(),
                    "agent_id": result.agentId,
                    "operation_type": result.operation_type,
                    "validation": result.validation.model_dump(),
                    "estimated_rows_affected": result.estimated_rows_affected,
                    "preview_data": result.preview_data
                }
            )

    def validate_payload(
        self,
        payload: DatabasePayload,
        prompt: str,
        attachments: List[str]
    ) -> List[ValidationIssue]:
        """
        Validate database payload

        For database operations, most validation is done by the Pydantic agents.
        This method adds additional safety checks.

        Args:
            payload: DatabasePayload to validate
            prompt: Original user prompt
            attachments: Attached documents

        Returns:
            List of validation issues (errors, warnings, info)
        """
        issues = []

        # Schema inspection - no validation needed
        if payload.operation == DatabaseOperationType.SCHEMA_INSPECTION:
            return issues

        # Query validation
        if payload.operation == DatabaseOperationType.QUERY:
            if not payload.sql_query:
                issues.append(ValidationIssue(
                    field="sql_query",
                    issue="No SQL query generated",
                    severity="error",
                    suggested_fix="Rephrase your question or check if tables exist",
                    auto_fixable=False
                ))
                return issues

            # Check for dangerous keywords in SELECT query
            dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
            query_upper = payload.sql_query.upper()
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    issues.append(ValidationIssue(
                        field="sql_query",
                        issue=f"Query contains forbidden keyword: {keyword}",
                        severity="error",
                        suggested_fix="Use SQL Execution for data modifications",
                        auto_fixable=False
                    ))

            return issues

        # Execution validation
        if payload.operation == DatabaseOperationType.EXECUTION:
            if not payload.sql_statements or len(payload.sql_statements) == 0:
                issues.append(ValidationIssue(
                    field="sql_statements",
                    issue="No SQL statements generated",
                    severity="error",
                    suggested_fix="Provide data or create table schema",
                    auto_fixable=False
                ))
                return issues

            # Extract validation issues from agent output
            agent_output = payload.metadata.get("agent_output", {})
            validation_data = agent_output.get("validation", {})
            agent_issues = validation_data.get("issues", [])

            for agent_issue in agent_issues:
                severity = agent_issue.get("severity", "info")
                issues.append(ValidationIssue(
                    field="sql_execution",
                    issue=agent_issue.get("message", "Unknown issue"),
                    severity=severity,
                    suggested_fix=agent_issue.get("suggestion"),
                    auto_fixable=False
                ))

            # Check for missing WHERE clause in DELETE/UPDATE
            for stmt in payload.sql_statements:
                stmt_upper = stmt.upper()
                if "DELETE FROM" in stmt_upper and "WHERE" not in stmt_upper:
                    issues.append(ValidationIssue(
                        field="sql_statements",
                        issue="⚠️ DELETE without WHERE clause will remove ALL rows!",
                        severity="error",
                        suggested_fix="Add WHERE clause to limit deletion",
                        auto_fixable=False
                    ))

                if "UPDATE" in stmt_upper and "WHERE" not in stmt_upper:
                    issues.append(ValidationIssue(
                        field="sql_statements",
                        issue="⚠️ UPDATE without WHERE clause will modify ALL rows!",
                        severity="warning",
                        suggested_fix="Add WHERE clause to limit updates",
                        auto_fixable=False
                    ))

            return issues

        return issues

    def execute(self, payload: DatabasePayload) -> ProviderResponse:
        """
        Execute database operation

        Note: For client-side DuckDB, actual execution happens in the browser.
        This method formats the response for the frontend to execute.

        Args:
            payload: DatabasePayload to execute

        Returns:
            ProviderResponse with checkpoint data for frontend execution
        """
        start_time = time.time()

        try:
            # Format response based on operation type
            if payload.operation == DatabaseOperationType.SCHEMA_INSPECTION:
                checkpoint_data = {
                    "checkpoint_type": "SCHEMA_INSPECTION",
                    "operation": "schema_inspection",
                    "auto_execute": True,
                    "requires_approval": False,
                    "agent_output": payload.metadata.get("agent_output", {})
                }

            elif payload.operation == DatabaseOperationType.QUERY:
                checkpoint_data = {
                    "checkpoint_type": "DATABASE_QUERY",
                    "operation": "query",
                    "query": payload.sql_query,
                    "description": payload.description,
                    "auto_execute": True,
                    "requires_approval": False,
                    "agent_output": payload.metadata.get("agent_output", {})
                }

            else:  # EXECUTION
                checkpoint_data = {
                    "checkpoint_type": "SQL_EXECUTION",
                    "operation": "execution",
                    "sql_statements": payload.sql_statements,
                    "description": payload.description,
                    "auto_execute": False,
                    "requires_approval": True,
                    "operation_type": payload.metadata.get("operation_type", "INSERT"),
                    "estimated_rows_affected": payload.metadata.get("estimated_rows_affected", 0),
                    "preview_data": payload.metadata.get("preview_data"),
                    "validation": payload.metadata.get("validation", {}),
                    "agent_output": payload.metadata.get("agent_output", {})
                }

            execution_time = int((time.time() - start_time) * 1000)

            return ProviderResponse(
                raw_response=checkpoint_data,
                processed_response=payload.description,
                metadata={
                    "checkpoint_data": checkpoint_data,
                    "operation": payload.operation.value,
                    "requires_approval": payload.requires_approval,
                    "auto_execute": payload.auto_execute,
                    "execution_time_ms": execution_time
                },
                execution_time_ms=execution_time,
                status_code=200
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            print(f"❌ Database provider execution error: {e}")

            return ProviderResponse(
                raw_response=None,
                processed_response="",
                metadata={
                    "error_details": {
                        "error_type": "database_error",
                        "recoverable": True,
                        "message": str(e)
                    },
                    "execution_time_ms": execution_time
                },
                execution_time_ms=execution_time,
                error=str(e),
                status_code=500
            )

    def audit_response(self, response: ProviderResponse) -> str:
        """
        Clean and audit database response for user consumption

        Args:
            response: ProviderResponse from execute()

        Returns:
            User-friendly message
        """
        if response.error:
            return f"❌ Database error: {response.error}"

        # Extract operation type
        metadata = response.metadata or {}
        checkpoint_data = metadata.get("checkpoint_data", {})
        operation = checkpoint_data.get("operation", "unknown")

        # Schema inspection
        if operation == "schema_inspection":
            agent_output = checkpoint_data.get("agent_output", {})
            return agent_output.get("message", "Schema inspection completed")

        # Query
        if operation == "query":
            description = checkpoint_data.get("description", "Query created")
            query = checkpoint_data.get("query", "")
            return f"📊 {description}\n\n```sql\n{query}\n```"

        # Execution
        if operation == "execution":
            description = checkpoint_data.get("description", "SQL operation ready")
            sql_statements = checkpoint_data.get("sql_statements", [])
            operation_type = checkpoint_data.get("operation_type", "SQL")

            sql_preview = "\n".join(sql_statements[:3])  # Show first 3 statements
            if len(sql_statements) > 3:
                sql_preview += f"\n... and {len(sql_statements) - 3} more"

            return f"⚙️ {operation_type} operation: {description}\n\n```sql\n{sql_preview}\n```\n\n⚠️ Requires approval before execution"

        # Fallback
        return response.processed_response or "Operation completed"
