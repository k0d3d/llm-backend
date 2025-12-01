"""
Database Query Agent

Pydantic AI agent for creating and validating SELECT queries.
Auto-executes without HITL approval since SELECT is read-only.
"""

from typing import List, Optional
from pydantic_ai import Agent
from pydantic import BaseModel, Field


class DatabaseQueryInput(BaseModel):
    """Input for database query agent"""
    user_request: str = Field(
        description="User's natural language request for data"
    )
    current_schema: dict = Field(
        default_factory=dict,
        description="Current database schema from frontend"
    )


class QueryValidation(BaseModel):
    """Query validation result"""
    is_valid: bool
    is_select_only: bool
    error: Optional[str] = None
    warning: Optional[str] = None


class DatabaseQueryOutput(BaseModel):
    """Output from database query agent"""
    status: str = Field(description="success | error")
    agentId: str = "database_query_agent"
    sql_query: str
    description: str
    validation: QueryValidation
    message: str
    auto_execute: bool = True
    requires_approval: bool = False


query_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=DatabaseQueryOutput,
    system_prompt="""You are a Database Query Agent for client-side DuckDB databases.

Your job:
1. Convert user's natural language requests into SQL SELECT queries
2. Validate queries are SELECT-only (no modifications allowed)
3. Ensure queries are syntactically correct
4. Use actual table/column names from the schema
5. Provide helpful descriptions of what the query does

Critical Rules:
1. ONLY generate SELECT queries - no INSERT/UPDATE/DELETE/CREATE/DROP/ALTER
2. Validate against actual schema - don't query non-existent tables/columns
3. Use proper SQL syntax for DuckDB
4. Include helpful WHERE/ORDER BY/LIMIT clauses when appropriate
5. Set auto_execute=true and requires_approval=false (SELECT is safe)

Supported Query Types:
- SELECT with WHERE filters
- SELECT with JOIN operations
- SELECT with GROUP BY aggregations
- SELECT with ORDER BY sorting
- SELECT with LIMIT pagination

Query Validation:
- Check for dangerous keywords (INSERT, UPDATE, DELETE, DROP, etc.)
- Verify tables exist in schema
- Verify columns exist in tables
- Check for syntax errors (unbalanced quotes, parentheses)

Example current_schema format:
{
  "tables": [
    {
      "name": "invoices",
      "rowCount": 3,
      "columns": [
        {"name": "id", "type": "BIGINT"},
        {"name": "invoice_number", "type": "VARCHAR"},
        {"name": "amount", "type": "DECIMAL"},
        {"name": "invoice_date", "type": "DATE"}
      ]
    },
    {
      "name": "customers",
      "rowCount": 5,
      "columns": [
        {"name": "id", "type": "BIGINT"},
        {"name": "name", "type": "VARCHAR"}
      ]
    }
  ]
}

Example Outputs:

For "show me all invoices":
{
  "status": "success",
  "agentId": "database_query_agent",
  "sql_query": "SELECT * FROM invoices ORDER BY invoice_date DESC",
  "description": "Retrieve all invoices sorted by date (newest first)",
  "validation": {
    "is_valid": true,
    "is_select_only": true,
    "error": null,
    "warning": null
  },
  "message": "Query created successfully. Will auto-execute.",
  "auto_execute": true,
  "requires_approval": false
}

For "total amount by customer":
{
  "status": "success",
  "agentId": "database_query_agent",
  "sql_query": "SELECT customer, SUM(amount) as total_amount FROM invoices GROUP BY customer ORDER BY total_amount DESC",
  "description": "Calculate total invoice amount for each customer",
  "validation": {
    "is_valid": true,
    "is_select_only": true,
    "error": null,
    "warning": null
  },
  "message": "Aggregation query created. Will auto-execute.",
  "auto_execute": true,
  "requires_approval": false
}

For invalid request (table doesn't exist):
{
  "status": "error",
  "agentId": "database_query_agent",
  "sql_query": "",
  "description": "",
  "validation": {
    "is_valid": false,
    "is_select_only": true,
    "error": "Table 'orders' does not exist in schema",
    "warning": null
  },
  "message": "Cannot create query: table 'orders' not found",
  "auto_execute": false,
  "requires_approval": false
}

For dangerous query:
{
  "status": "error",
  "agentId": "database_query_agent",
  "sql_query": "",
  "description": "",
  "validation": {
    "is_valid": false,
    "is_select_only": false,
    "error": "Only SELECT queries allowed. Use SQL Execution Tool for modifications.",
    "warning": null
  },
  "message": "Rejected: query contains forbidden operations",
  "auto_execute": false,
  "requires_approval": false
}
""",
)


async def create_database_query(
    user_request: str,
    current_schema: Optional[dict] = None
) -> DatabaseQueryOutput:
    """
    Create and validate a SELECT query using AI agent

    Args:
        user_request: User's natural language request for data
        current_schema: Current database schema from frontend

    Returns:
        DatabaseQueryOutput with SQL query and validation
    """

    if current_schema is None:
        current_schema = {"tables": []}

    input_data = DatabaseQueryInput(
        user_request=user_request,
        current_schema=current_schema
    )

    # Build context for the agent
    tables = current_schema.get('tables', [])
    schema_summary = []
    for table in tables:
        columns = table.get('columns', [])
        col_names = [col.get('name') for col in columns]
        row_count = table.get('rowCount', 0)
        schema_summary.append(
            f"  - {table.get('name')} ({row_count} rows): {', '.join(col_names)}"
        )

    schema_summary_str = "\n".join(schema_summary) if schema_summary else "  (no tables)"

    prompt = f"""Create a SQL SELECT query for this user request: "{user_request}"

AVAILABLE DATABASE SCHEMA:
{schema_summary_str}

FULL SCHEMA DATA:
{current_schema}

Task:
1. Convert the user's request into a valid SQL SELECT query
2. Use ONLY tables and columns that exist in the schema above
3. Validate the query is SELECT-only (no modifications)
4. Check for syntax errors
5. Provide a clear description of what the query does
6. Set appropriate validation flags

User Request: "{user_request}"

Remember:
- ONLY SELECT allowed - reject any INSERT/UPDATE/DELETE/CREATE/DROP/ALTER
- Use actual table/column names from the schema
- Include helpful WHERE/ORDER BY/LIMIT clauses
- Provide clear error messages if request cannot be fulfilled
"""

    try:
        result = await query_agent.run(prompt, deps=input_data)
        return result.output
    except Exception as agent_error:
        print(f"⚠️ Database query agent failed: {agent_error}")
        # Fallback to error response
        return DatabaseQueryOutput(
            status="error",
            agentId="database_query_agent",
            sql_query="",
            description="",
            validation=QueryValidation(
                is_valid=False,
                is_select_only=True,
                error=f"Query generation failed: {str(agent_error)}"
            ),
            message=f"Failed to create query: {str(agent_error)}",
            auto_execute=False,
            requires_approval=False
        )
