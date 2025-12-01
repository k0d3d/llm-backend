"""
Schema Inspection Agent

Pydantic AI agent for inspecting client-side DuckDB database schemas.
Handles "show tables", "describe TABLE_NAME" operations without HITL approval.
"""

from typing import List, Optional
from pydantic_ai import Agent
from pydantic import BaseModel, Field


class SchemaInspectionInput(BaseModel):
    """Input for schema inspection agent"""
    query: str = Field(
        description="Schema query: 'show tables', 'describe TABLE_NAME', 'tables'"
    )
    current_schema: dict = Field(
        default_factory=dict,
        description="Current database schema from frontend"
    )


class ColumnInfo(BaseModel):
    """Column information"""
    name: str
    type: str
    pk: bool = False
    nullable: bool = True


class TableInfo(BaseModel):
    """Table information"""
    name: str
    rowCount: int = 0
    columns: List[ColumnInfo] = Field(default_factory=list)


class SchemaInspectionOutput(BaseModel):
    """Output from schema inspection agent"""
    status: str = Field(description="success | error")
    agentId: str = "database_schema_agent"
    tableCount: int = 0
    tables: List[TableInfo] = Field(default_factory=list)
    message: str
    error: Optional[str] = None


schema_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=SchemaInspectionOutput,
    system_prompt="""You are a Database Schema Inspection Agent for client-side DuckDB databases.

Your job:
1. Parse schema inspection requests from users
2. Return structured table/column information from the current schema
3. Include row counts for each table
4. Provide clear, helpful messages
5. Handle errors gracefully with informative error messages

Supported Operations:
- "show tables" → List all tables with row counts
- "tables" → Same as show tables
- "describe {table}" → Show columns for specific table
- "schema" → Show complete schema information

Critical Rules:
1. ALWAYS return valid JSON matching SchemaInspectionOutput model
2. Extract table/column info from current_schema provided in context
3. Set status="success" for valid requests, status="error" for failures
4. Include helpful messages explaining what data is being shown
5. For "describe" queries, filter to the specific table requested

Example current_schema format:
{
  "tables": [
    {
      "name": "invoices",
      "rowCount": 3,
      "columns": [
        {"name": "id", "type": "BIGINT", "pk": true},
        {"name": "customer", "type": "VARCHAR"}
      ]
    }
  ]
}

Example Outputs:

For "show tables":
{
  "status": "success",
  "agentId": "database_schema_agent",
  "tableCount": 2,
  "tables": [
    {
      "name": "invoices",
      "rowCount": 3,
      "columns": [...]
    },
    {
      "name": "customers",
      "rowCount": 0,
      "columns": [...]
    }
  ],
  "message": "Found 2 tables in database: invoices (3 rows), customers (0 rows)"
}

For "describe invoices":
{
  "status": "success",
  "agentId": "database_schema_agent",
  "tableCount": 1,
  "tables": [
    {
      "name": "invoices",
      "rowCount": 3,
      "columns": [
        {"name": "id", "type": "BIGINT", "pk": true},
        {"name": "customer", "type": "VARCHAR"}
      ]
    }
  ],
  "message": "Table 'invoices' has 2 columns and 3 rows"
}

For errors:
{
  "status": "error",
  "agentId": "database_schema_agent",
  "tableCount": 0,
  "tables": [],
  "message": "Table 'unknown_table' not found in database",
  "error": "Table not found"
}
""",
)


async def inspect_schema(
    query: str,
    current_schema: Optional[dict] = None
) -> SchemaInspectionOutput:
    """
    Inspect database schema using AI agent

    Args:
        query: Schema inspection query (e.g., "show tables", "describe invoices")
        current_schema: Current database schema from frontend

    Returns:
        SchemaInspectionOutput with table/column information
    """

    if current_schema is None:
        current_schema = {"tables": []}

    input_data = SchemaInspectionInput(
        query=query,
        current_schema=current_schema
    )

    # Build context for the agent
    tables = current_schema.get('tables', [])
    table_summary = []
    for table in tables:
        row_count = table.get('rowCount', 0)
        col_count = len(table.get('columns', []))
        table_summary.append(f"  - {table.get('name')}: {col_count} columns, {row_count} rows")

    table_summary_str = "\n".join(table_summary) if table_summary else "  (no tables)"

    prompt = f"""Inspect database schema for this query: "{query}"

CURRENT DATABASE SCHEMA:
Tables: {len(tables)}
{table_summary_str}

FULL SCHEMA DATA:
{current_schema}

Task:
1. Parse the user's query to understand what they want to see
2. Extract relevant information from the current_schema
3. Format it according to the SchemaInspectionOutput model
4. Provide a helpful message summarizing the results

Query: "{query}"
"""

    try:
        result = await schema_agent.run(prompt, deps=input_data)
        return result.output
    except Exception as agent_error:
        print(f"⚠️ Schema inspection agent failed: {agent_error}")
        # Fallback to basic schema return
        return SchemaInspectionOutput(
            status="error",
            agentId="database_schema_agent",
            tableCount=0,
            tables=[],
            message=f"Schema inspection failed: {str(agent_error)}",
            error=str(agent_error)
        )
