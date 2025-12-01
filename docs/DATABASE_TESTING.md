# Database Migration Testing Guide

This document provides testing instructions for the CLIENT_DATABASE Pydantic AI migration.

## What Was Implemented

### 1. Pydantic AI Agents (✅ Complete)
- **`database_schema_agent.py`**: Schema inspection (show tables, describe)
- **`database_query_agent.py`**: SELECT query creation (auto-execute)
- **`database_execution_agent.py`**: INSERT/UPDATE/DELETE with HITL approval

### 2. DatabaseProvider (✅ Complete)
- **`database_provider.py`**: Follows AIProvider interface pattern
- Implements: `get_capabilities()`, `create_payload()`, `validate_payload()`, `execute()`, `audit_response()`
- Routes to appropriate Pydantic agent based on user intent

### 3. API Endpoints (✅ Complete)
- **`/api/database/run`**: Main endpoint for database operations
- **`/api/database/approve`**: Approval endpoint for SQL execution
- **`/api/database/run/{run_id}`**: Status endpoint
- **`/api/database/health`**: Health check

### 4. CF Workers Routing (✅ Complete)
- Updated `AgentRoutingService.ts` to route CLIENT_DATABASE to `PY_LLM_API_URL/api/database/run`
- Separated from other CrewAI tools (RSS, Web3, Woo, Prompt)

### 5. Provider Registry (✅ Complete)
- Added `AgentTools.CLIENT_DATABASE` enum
- Added database operation types to `OperationType` enum
- Registered `DatabaseProvider` in `registry_setup.py`

---

## Architecture Flow

```
User Request
    ↓
CF Workers (AgentRoutingService.ts)
    ↓ (CLIENT_DATABASE detected)
    ↓
llm-backend: /api/database/run
    ↓
DatabaseProvider.create_payload()
    ↓
┌─────────────────────────────────────┐
│ Detect Operation Type:              │
│ - Schema Inspection                 │
│ - Query (SELECT)                    │
│ - Execution (INSERT/UPDATE/DELETE)  │
└─────────────────────────────────────┘
    ↓
Route to Pydantic AI Agent:
    ↓
┌───────────────────────┬───────────────────────┬─────────────────────────┐
│ SchemaInspectionAgent │ DatabaseQueryAgent    │ SQLExecutionAgent       │
│ (read-only)           │ (SELECT only)         │ (write operations)      │
│ Auto-execute          │ Auto-execute          │ Requires HITL approval  │
└───────────────────────┴───────────────────────┴─────────────────────────┘
    ↓
DatabaseProvider.validate_payload()
    ↓
DatabaseProvider.execute()
    ↓
┌─────────────────────────────────────┐
│ Return checkpoint_data for frontend │
│ - auto_execute: true/false          │
│ - requires_approval: true/false     │
│ - sql_query or sql_statements       │
└─────────────────────────────────────┘
    ↓
Frontend executes in DuckDB-WASM
```

---

## Testing Checklist

### ✅ Unit Tests (Code Compilation)
- [x] `database_schema_agent.py` - No syntax errors
- [x] `database_query_agent.py` - No syntax errors
- [x] `database_execution_agent.py` - No syntax errors
- [x] `database_provider.py` - No syntax errors
- [x] `database.py` (endpoints) - No syntax errors

### 🧪 Integration Tests (Manual)

#### Test 1: Schema Inspection
**Request:**
```bash
POST /api/database/run
{
  "prompt": "show tables",
  "session_id": "test-session-001",
  "user_id": "user-123",
  "agent_tool_config": {
    "client-database-tool": {
      "current_schema": {
        "tables": [
          {"name": "invoices", "rowCount": 3, "columns": [...]},
          {"name": "customers", "rowCount": 0, "columns": [...]}
        ]
      }
    }
  }
}
```

**Expected Response:**
```json
{
  "run_id": "...",
  "status": "completed",
  "operation_type": "schema_inspection",
  "requires_approval": false,
  "auto_execute": true,
  "checkpoint_data": {
    "checkpoint_type": "SCHEMA_INSPECTION",
    "agent_output": {
      "status": "success",
      "tableCount": 2,
      "tables": [...]
    }
  }
}
```

#### Test 2: Database Query (SELECT)
**Request:**
```bash
POST /api/database/run
{
  "prompt": "show me all invoices",
  "session_id": "test-session-002",
  "user_id": "user-123",
  "agent_tool_config": {
    "client-database-tool": {
      "current_schema": {
        "tables": [
          {
            "name": "invoices",
            "rowCount": 3,
            "columns": [
              {"name": "id", "type": "BIGINT"},
              {"name": "customer", "type": "VARCHAR"},
              {"name": "amount", "type": "DECIMAL"}
            ]
          }
        ]
      }
    }
  }
}
```

**Expected Response:**
```json
{
  "run_id": "...",
  "status": "completed",
  "operation_type": "query",
  "requires_approval": false,
  "auto_execute": true,
  "checkpoint_data": {
    "checkpoint_type": "DATABASE_QUERY",
    "query": "SELECT * FROM invoices ORDER BY id",
    "description": "Retrieve all invoices",
    "auto_execute": true
  }
}
```

#### Test 3: SQL Execution (INSERT with HITL)
**Request:**
```bash
POST /api/database/run
{
  "prompt": "insert invoice data",
  "session_id": "test-session-003",
  "user_id": "user-123",
  "agent_tool_config": {
    "client-database-tool": {
      "current_schema": {
        "tables": [
          {
            "name": "invoices",
            "rowCount": 3,
            "columns": [
              {"name": "id", "type": "BIGINT"},
              {"name": "invoice_number", "type": "VARCHAR"},
              {"name": "customer", "type": "VARCHAR"},
              {"name": "amount", "type": "DECIMAL"}
            ]
          }
        ]
      }
    }
  },
  "document_data": {
    "invoice_number": "INV-001",
    "customer": "Acme Corp",
    "amount": 1500.00
  }
}
```

**Expected Response:**
```json
{
  "run_id": "...",
  "status": "waiting_approval",
  "operation_type": "execution",
  "requires_approval": true,
  "auto_execute": false,
  "checkpoint_data": {
    "checkpoint_type": "SQL_EXECUTION",
    "operation_type": "INSERT",
    "sql_statements": [
      "INSERT INTO invoices (invoice_number, customer, amount) VALUES ('INV-001', 'Acme Corp', 1500.00)"
    ],
    "description": "Insert invoice INV-001",
    "requires_approval": true,
    "validation": {
      "is_valid": true,
      "blocking_issues": 0
    }
  },
  "websocket_url": "ws://..."
}
```

**Approval Request:**
```bash
POST /api/database/approve
{
  "run_id": "...",
  "session_id": "test-session-003",
  "approved": true
}
```

---

## End-to-End Flow Test

### Prerequisites
1. llm-backend server running: `poetry run fastapi dev src/main.py`
2. CF Workers deployed with updated routing
3. Frontend with DuckDB-WASM ready to execute SQL

### Test Scenario: Invoice PDF → Database Insert

**Step 1: User uploads PDF invoice**
- Frontend sends request to CF Workers
- CF Workers routes to `/api/database/run`

**Step 2: DatabaseProvider analyzes request**
- Detects operation type: EXECUTION (INSERT)
- Routes to `SQLExecutionAgent`

**Step 3: SQLExecutionAgent extracts data**
- Parses `document_data` from PDF
- Generates INSERT statements
- Returns validation (blocking_issues: 0)

**Step 4: HITL Checkpoint**
- Frontend displays SQL preview
- Shows 10-second auto-approve countdown (if no blocking issues)
- User can approve/reject/edit

**Step 5: Frontend executes in DuckDB**
- Runs SQL in browser
- Returns results to user

**Step 6: Subsequent query**
- User asks: "show me all invoices"
- `DatabaseQueryAgent` creates SELECT query
- Auto-executes immediately (no approval)
- Returns invoice data including newly inserted row

---

## Verification Commands

### Check Provider Registration
```python
from llm_backend.core.providers.registry import ProviderRegistry

# List all providers
print(ProviderRegistry.list_providers())
# Expected: ['replicate', 'database']

# List all tools
print(ProviderRegistry.list_tools())
# Expected: [<AgentTools.REPLICATETOOL>, <AgentTools.CLIENT_DATABASE>]
```

### Test Agent Directly
```python
import asyncio
from llm_backend.agents.database_schema_agent import inspect_schema

current_schema = {
    "tables": [
        {"name": "invoices", "rowCount": 3, "columns": []},
        {"name": "customers", "rowCount": 0, "columns": []}
    ]
}

result = asyncio.run(inspect_schema("show tables", current_schema))
print(result)
# Expected: SchemaInspectionOutput with status="success", tableCount=2
```

### Test Provider Creation
```python
from llm_backend.providers.database_provider import DatabaseProvider

provider = DatabaseProvider(config={
    "current_schema": {"tables": []}
})

capabilities = provider.get_capabilities()
print(capabilities.name)  # Expected: "DuckDB Client-Side Database"
print(capabilities.supported_operations)  # Expected: [DATABASE_QUERY, DATABASE_WRITE, DATABASE_SCHEMA]
```

---

## Known Issues & Limitations

### Current Implementation
✅ **Type-safe**: Pydantic models enforce schema
✅ **HITL integration**: SQL execution requires approval
✅ **Auto-execute**: Schema/query operations run immediately
✅ **Client-side**: DuckDB-WASM executes in browser (no server DB)

### Future Enhancements
- [ ] PDF extraction agent (currently uses `document_data` from frontend)
- [ ] Multi-table INSERT transactions
- [ ] Schema migration tracking
- [ ] Query result caching
- [ ] Database export/import

---

## Troubleshooting

### Error: "No provider registered for tool: CLIENT_DATABASE"
**Solution:** Ensure `registry_setup.py` is imported before creating orchestrator
```python
from llm_backend.providers import registry_setup  # This auto-registers providers
```

### Error: "Agent did not return an output payload"
**Solution:** Check Pydantic agent system prompts and ensure `output_type` is set correctly

### Error: "Only SELECT queries allowed"
**Solution:** User is trying to use `DatabaseQueryAgent` for write operations. Request should route to `SQLExecutionAgent` instead.

---

## Migration Comparison

### Before (CrewAI in tohju-py-api)
- ❌ String-based tool calls → format errors
- ❌ Agent hallucination after tool failures
- ❌ No type safety
- ❌ PDF re-downloaded on every request
- ❌ Tools wrapped inputs in arrays

### After (Pydantic AI in llm-backend)
- ✅ Type-safe Pydantic models
- ✅ Automatic validation
- ✅ Clear error messages
- ✅ HITL integration
- ✅ No PDF re-download (handled by frontend)
- ✅ Properly typed inputs/outputs

---

## Success Criteria

The migration is successful if:

1. ✅ **Schema inspection works**: "show tables" returns table list
2. ✅ **Queries auto-execute**: "show me invoices" runs immediately
3. ✅ **INSERT requires approval**: "insert data" creates HITL checkpoint
4. ✅ **No format errors**: Agent inputs/outputs match Pydantic schemas
5. ✅ **No hallucination**: Failed operations return proper error messages
6. ✅ **CF Workers routing works**: CLIENT_DATABASE requests go to llm-backend
7. ✅ **End-to-end flow**: PDF → INSERT approval → query → results

---

## Next Steps

After successful testing:

1. Update frontend to handle new checkpoint formats
2. Add WebSocket listener for approval flow
3. Implement PDF extraction in llm-backend (if needed)
4. Add telemetry/logging for database operations
5. Create integration tests for CI/CD
6. Document API for frontend team
