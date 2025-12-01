"""
SQL Execution Agent

Pydantic AI agent for creating and validating INSERT/UPDATE/DELETE/CREATE queries.
Requires HITL approval before execution since these operations modify data.
"""

from typing import List, Optional, Dict, Any
from pydantic_ai import Agent
from pydantic import BaseModel, Field


class SQLExecutionInput(BaseModel):
    """Input for SQL execution agent"""
    user_request: str = Field(
        description="User's natural language request for data modification"
    )
    operation_type: str = Field(
        description="Type of operation: INSERT, UPDATE, DELETE, CREATE, ALTER"
    )
    current_schema: dict = Field(
        default_factory=dict,
        description="Current database schema from frontend"
    )
    document_data: Optional[dict] = Field(
        default=None,
        description="Extracted data from PDF/document for INSERT operations"
    )


class ValidationIssue(BaseModel):
    """Validation issue for SQL execution"""
    severity: str = Field(description="error | warning | info")
    message: str
    suggestion: Optional[str] = None


class SQLExecutionValidation(BaseModel):
    """SQL execution validation result"""
    is_valid: bool
    is_safe: bool
    blocking_issues: int = 0
    issues: List[ValidationIssue] = Field(default_factory=list)


class SQLExecutionOutput(BaseModel):
    """Output from SQL execution agent"""
    status: str = Field(description="success | error")
    agentId: str = "database_execution_agent"
    operation_type: str = Field(description="INSERT | UPDATE | DELETE | CREATE | ALTER")
    sql_statements: List[str] = Field(
        description="SQL statements to execute (can be multiple for transactions)"
    )
    description: str
    validation: SQLExecutionValidation
    estimated_rows_affected: int = 0
    message: str
    auto_execute: bool = False
    requires_approval: bool = True
    preview_data: Optional[Dict[str, Any]] = None


execution_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=SQLExecutionOutput,
    system_prompt="""You are a SQL Execution Agent for client-side DuckDB databases.

Your job:
1. Convert user's natural language requests into SQL modification queries
2. Generate INSERT/UPDATE/DELETE/CREATE/ALTER statements
3. Validate queries are safe and correct
4. Extract data from documents (PDFs) for INSERT operations
5. Provide clear descriptions and warnings
6. ALWAYS require human approval before execution

Critical Rules:
1. ALWAYS set requires_approval=true and auto_execute=false
2. Generate valid SQL for INSERT/UPDATE/DELETE/CREATE/ALTER operations
3. Validate against actual schema - don't modify non-existent tables
4. For INSERT: extract data from document_data if provided
5. For CREATE: generate proper DDL based on data structure
6. Provide clear warnings about data modifications
7. Estimate number of rows that will be affected

Supported Operations:

**INSERT:**
- Extract data from document_data (PDF invoice, CSV, etc.)
- Generate CREATE TABLE if table doesn't exist
- Generate INSERT statements with actual data
- Use transactions for multiple inserts

**UPDATE:**
- Validate WHERE clause to prevent accidental mass updates
- Warn if WHERE clause is missing (affects all rows)
- Show preview of what will be updated

**DELETE:**
- Validate WHERE clause to prevent accidental mass deletes
- Warn if WHERE clause is missing (deletes all rows!)
- Require confirmation for destructive operations

**CREATE/ALTER:**
- Generate proper DDL syntax
- Validate column types
- Handle constraints (PRIMARY KEY, NOT NULL, etc.)

Validation Severity:
- **error**: Blocks execution (invalid SQL, missing data, safety issue)
- **warning**: Allows execution but user should review (no WHERE clause, large batch)
- **info**: Informational (table will be created, rows will be affected)

Example current_schema format:
{
  "tables": [
    {
      "name": "invoices",
      "rowCount": 3,
      "columns": [
        {"name": "id", "type": "BIGINT", "pk": true},
        {"name": "invoice_number", "type": "VARCHAR"},
        {"name": "amount", "type": "DECIMAL"},
        {"name": "invoice_date", "type": "DATE"}
      ]
    }
  ]
}

Example document_data format (from PDF):
{
  "invoice_number": "INV-001",
  "customer": "Acme Corp",
  "amount": 1500.00,
  "invoice_date": "2025-01-15",
  "items": [
    {"description": "Service A", "quantity": 2, "price": 500.00},
    {"description": "Service B", "quantity": 1, "price": 500.00}
  ]
}

Example Outputs:

For "insert invoice data" (with document_data):
{
  "status": "success",
  "agentId": "database_execution_agent",
  "operation_type": "INSERT",
  "sql_statements": [
    "INSERT INTO invoices (invoice_number, customer, amount, invoice_date) VALUES ('INV-001', 'Acme Corp', 1500.00, '2025-01-15')",
    "INSERT INTO invoice_items (invoice_id, description, quantity, price) VALUES (LAST_INSERT_ID(), 'Service A', 2, 500.00)",
    "INSERT INTO invoice_items (invoice_id, description, quantity, price) VALUES (LAST_INSERT_ID(), 'Service B', 1, 500.00)"
  ],
  "description": "Insert invoice INV-001 with 2 line items",
  "validation": {
    "is_valid": true,
    "is_safe": true,
    "blocking_issues": 0,
    "issues": [
      {
        "severity": "info",
        "message": "Will insert 1 invoice and 2 line items",
        "suggestion": null
      }
    ]
  },
  "estimated_rows_affected": 3,
  "message": "Ready to insert invoice data. Requires approval.",
  "auto_execute": false,
  "requires_approval": true,
  "preview_data": {
    "invoice": {"invoice_number": "INV-001", "customer": "Acme Corp", "amount": 1500.00},
    "items": 2
  }
}

For "delete all invoices" (dangerous!):
{
  "status": "success",
  "agentId": "database_execution_agent",
  "operation_type": "DELETE",
  "sql_statements": [
    "DELETE FROM invoices"
  ],
  "description": "Delete ALL invoices from database",
  "validation": {
    "is_valid": true,
    "is_safe": false,
    "blocking_issues": 1,
    "issues": [
      {
        "severity": "error",
        "message": "⚠️ DANGER: No WHERE clause - this will delete ALL 3 rows!",
        "suggestion": "Add WHERE clause to limit deletion, or confirm you want to delete all data"
      }
    ]
  },
  "estimated_rows_affected": 3,
  "message": "⚠️ DESTRUCTIVE OPERATION - Review carefully before approving",
  "auto_execute": false,
  "requires_approval": true,
  "preview_data": null
}

For "create invoices table" (table doesn't exist):
{
  "status": "success",
  "agentId": "database_execution_agent",
  "operation_type": "CREATE",
  "sql_statements": [
    "CREATE TABLE invoices (id BIGINT PRIMARY KEY, invoice_number VARCHAR, customer VARCHAR, amount DECIMAL, invoice_date DATE)"
  ],
  "description": "Create new 'invoices' table with 5 columns",
  "validation": {
    "is_valid": true,
    "is_safe": true,
    "blocking_issues": 0,
    "issues": [
      {
        "severity": "info",
        "message": "Will create new table 'invoices'",
        "suggestion": null
      }
    ]
  },
  "estimated_rows_affected": 0,
  "message": "Ready to create table. Requires approval.",
  "auto_execute": false,
  "requires_approval": true,
  "preview_data": {
    "table_name": "invoices",
    "columns": 5
  }
}

For invalid operation (table doesn't exist, no document data):
{
  "status": "error",
  "agentId": "database_execution_agent",
  "operation_type": "INSERT",
  "sql_statements": [],
  "description": "",
  "validation": {
    "is_valid": false,
    "is_safe": false,
    "blocking_issues": 2,
    "issues": [
      {
        "severity": "error",
        "message": "Table 'invoices' does not exist",
        "suggestion": "Create table first or provide document data to auto-generate schema"
      },
      {
        "severity": "error",
        "message": "No document data provided for INSERT",
        "suggestion": "Upload a PDF/CSV file or provide data manually"
      }
    ]
  },
  "estimated_rows_affected": 0,
  "message": "Cannot create INSERT: table missing and no data provided",
  "auto_execute": false,
  "requires_approval": true,
  "preview_data": null
}
""",
)


async def create_sql_execution(
    user_request: str,
    operation_type: str,
    current_schema: Optional[dict] = None,
    document_data: Optional[dict] = None
) -> SQLExecutionOutput:
    """
    Create and validate SQL execution statements using AI agent

    Args:
        user_request: User's natural language request for data modification
        operation_type: Type of operation (INSERT, UPDATE, DELETE, CREATE, ALTER)
        current_schema: Current database schema from frontend
        document_data: Extracted data from PDF/document for INSERT operations

    Returns:
        SQLExecutionOutput with SQL statements and validation
    """

    if current_schema is None:
        current_schema = {"tables": []}

    input_data = SQLExecutionInput(
        user_request=user_request,
        operation_type=operation_type,
        current_schema=current_schema,
        document_data=document_data
    )

    # Build context for the agent
    tables = current_schema.get('tables', [])
    schema_summary = []
    for table in tables:
        columns = table.get('columns', [])
        col_info = []
        for col in columns:
            col_type = col.get('type', 'VARCHAR')
            col_name = col.get('name')
            pk = " (PK)" if col.get('pk', False) else ""
            col_info.append(f"{col_name} {col_type}{pk}")
        row_count = table.get('rowCount', 0)
        schema_summary.append(
            f"  - {table.get('name')} ({row_count} rows): {', '.join(col_info)}"
        )

    schema_summary_str = "\n".join(schema_summary) if schema_summary else "  (no tables)"

    doc_data_summary = "No document data provided"
    if document_data:
        doc_data_summary = f"Document data available with {len(document_data)} fields"

    prompt = f"""Create SQL {operation_type} statement(s) for this user request: "{user_request}"

OPERATION TYPE: {operation_type}

AVAILABLE DATABASE SCHEMA:
{schema_summary_str}

DOCUMENT DATA:
{doc_data_summary}
{document_data if document_data else "(none)"}

FULL SCHEMA DATA:
{current_schema}

Task:
1. Convert the user's request into valid SQL {operation_type} statement(s)
2. For INSERT: extract data from document_data if available, or generate CREATE TABLE + INSERT if table missing
3. For UPDATE/DELETE: validate WHERE clauses to prevent accidental mass modifications
4. For CREATE: generate proper DDL with appropriate column types
5. Validate the operation is safe and provide clear warnings
6. Estimate how many rows will be affected
7. ALWAYS set requires_approval=true and auto_execute=false

User Request: "{user_request}"

Remember:
- ALWAYS require approval (requires_approval=true, auto_execute=false)
- Validate against actual schema
- For INSERT: use document_data to populate values
- For DELETE/UPDATE: warn about missing WHERE clauses
- Provide clear descriptions and validation issues
- Mark blocking issues with severity="error"
"""

    try:
        result = await execution_agent.run(prompt, deps=input_data)
        return result.output
    except Exception as agent_error:
        print(f"⚠️ SQL execution agent failed: {agent_error}")
        # Fallback to error response
        return SQLExecutionOutput(
            status="error",
            agentId="database_execution_agent",
            operation_type=operation_type,
            sql_statements=[],
            description="",
            validation=SQLExecutionValidation(
                is_valid=False,
                is_safe=False,
                blocking_issues=1,
                issues=[ValidationIssue(
                    severity="error",
                    message=f"SQL generation failed: {str(agent_error)}",
                    suggestion="Check your request and try again"
                )]
            ),
            estimated_rows_affected=0,
            message=f"Failed to create SQL: {str(agent_error)}",
            auto_execute=False,
            requires_approval=True
        )
