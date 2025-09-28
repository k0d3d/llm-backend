#!/usr/bin/env python3
"""
Comprehensive HITL test runner

This script runs all HITL tests and generates a detailed report
focusing on session resumability and data persistence.
"""

import os
import sys
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path

# Add the project root and src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

def run_test_suite(test_file: str, description: str) -> dict:
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test suite
        )
        
        return {
            "name": description,
            "file": test_file,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "name": description,
            "file": test_file,
            "success": False,
            "stdout": "",
            "stderr": "Test suite timed out after 5 minutes",
            "returncode": -1
        }
    except Exception as e:
        return {
            "name": description,
            "file": test_file,
            "success": False,
            "stdout": "",
            "stderr": f"Failed to run test: {str(e)}",
            "returncode": -2
        }

def generate_report(results: list) -> str:
    """Generate a comprehensive test report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
HITL Test Report
Generated: {timestamp}

{'='*80}
SUMMARY
{'='*80}

Total Test Suites: {len(results)}
Passed: {sum(1 for r in results if r['success'])}
Failed: {sum(1 for r in results if not r['success'])}

"""
    
    # Add detailed results for each test suite
    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        report += f"""
{'-'*60}
{result['name']} - {status}
{'-'*60}
File: {result['file']}
Return Code: {result['returncode']}

"""
        
        if result['stdout']:
            report += f"STDOUT:\n{result['stdout']}\n\n"
        
        if result['stderr']:
            report += f"STDERR:\n{result['stderr']}\n\n"
    
    return report

async def check_database_state():
    """Check if database state properly reflects checkpoints for resumability"""
    print("\n" + "="*60)
    print("CHECKING DATABASE STATE FOR RESUMABILITY")
    print("="*60)
    
    try:
        from llm_backend.core.hitl.persistence import DatabaseStateStore
        from llm_backend.core.hitl.types import HITLState, HITLConfig, HITLStep, HITLStatus
        import uuid
        
        # Create test database store
        db_store = DatabaseStateStore("sqlite:///:memory:")
        
        # Create a test state that represents a paused HITL run
        run_id = str(uuid.uuid4())
        test_state = HITLState(
            run_id=run_id,
            current_step=HITLStep.INFORMATION_REVIEW,
            status=HITLStatus.AWAITING_HUMAN,
            config=HITLConfig(),
            original_input={
                "prompt": "Remove background from image",
                "session_id": "test_resumability_session",
                "user_id": "test_user_123"
            },
            validation_issues=[
                {
                    "field": "input_image",
                    "severity": "error",
                    "issue": "Required parameter 'input_image' is missing or empty"
                }
            ],
            pending_actions=["upload_image", "approve"],
            checkpoint_context={
                "type": "file_validation",
                "context": {
                    "requires_file_input": True,
                    "file_type": "image",
                    "user_friendly_message": "Please upload an image file"
                }
            }
        )
        
        # Save state
        await db_store.save_state(test_state)
        
        # Load state and verify resumability data
        loaded_state = await db_store.load_state(run_id)
        
        checks = []
        
        # Check 1: Basic state persistence
        checks.append({
            "name": "Basic state persistence",
            "passed": loaded_state is not None and loaded_state.run_id == run_id,
            "details": f"State loaded: {loaded_state is not None}"
        })
        
        # Check 2: Session/user data preservation
        checks.append({
            "name": "Session/user data preservation",
            "passed": (loaded_state.original_input.get("session_id") == "test_resumability_session" and
                      loaded_state.original_input.get("user_id") == "test_user_123"),
            "details": f"Session ID: {loaded_state.original_input.get('session_id')}, User ID: {loaded_state.original_input.get('user_id')}"
        })
        
        # Check 3: Checkpoint context preservation
        checks.append({
            "name": "Checkpoint context preservation",
            "passed": (loaded_state.checkpoint_context is not None and
                      loaded_state.checkpoint_context.get("type") == "file_validation"),
            "details": f"Checkpoint type: {loaded_state.checkpoint_context.get('type') if loaded_state.checkpoint_context else 'None'}"
        })
        
        # Check 4: Validation issues preservation
        checks.append({
            "name": "Validation issues preservation",
            "passed": (len(loaded_state.validation_issues) > 0 and
                      loaded_state.validation_issues[0].get("field") == "input_image"),
            "details": f"Validation issues count: {len(loaded_state.validation_issues)}"
        })
        
        # Check 5: Pending actions preservation
        checks.append({
            "name": "Pending actions preservation",
            "passed": ("upload_image" in loaded_state.pending_actions and
                      "approve" in loaded_state.pending_actions),
            "details": f"Pending actions: {loaded_state.pending_actions}"
        })
        
        # Check 6: Session-based query functionality
        session_runs = await db_store.list_active_runs(session_id="test_resumability_session")
        checks.append({
            "name": "Session-based query functionality",
            "passed": len(session_runs) == 1 and session_runs[0]["run_id"] == run_id,
            "details": f"Session runs found: {len(session_runs)}"
        })
        
        # Print results
        for check in checks:
            status = "✅ PASS" if check["passed"] else "❌ FAIL"
            print(f"{status} {check['name']}")
            print(f"    Details: {check['details']}")
        
        all_passed = all(check["passed"] for check in checks)
        
        if all_passed:
            print("\n🎉 All database state checks PASSED! Database properly reflects resumability state.")
        else:
            print("\n⚠️  Some database state checks FAILED! Resumability may be compromised.")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Database state check failed with error: {str(e)}")
        return False

def main():
    """Main test runner"""
    print("HITL Test Suite Runner")
    print("=" * 80)
    print("Testing session resumability, data persistence, and attachment handling")
    
    # Define test suites
    test_suites = [
        {
            "file": "tests/test_hitl_session_resumability.py",
            "description": "Session Resumability Tests"
        },
        {
            "file": "tests/test_hitl_database_integrity.py", 
            "description": "Database Integrity Tests"
        },
        {
            "file": "tests/test_hitl_edge_cases.py",
            "description": "Edge Cases and Error Handling Tests"
        }
    ]
    
    # Run test suites
    results = []
    for suite in test_suites:
        result = run_test_suite(suite["file"], suite["description"])
        results.append(result)
    
    # Run database state check
    print("\n" + "="*80)
    print("RUNNING DATABASE STATE VERIFICATION")
    print("="*80)
    
    try:
        db_check_passed = asyncio.run(check_database_state())
    except Exception as e:
        print(f"❌ Database state check failed: {str(e)}")
        db_check_passed = False
    
    # Generate and display report
    report = generate_report(results)
    print(report)
    
    # Add database check to summary
    print("="*80)
    print("DATABASE STATE VERIFICATION")
    print("="*80)
    if db_check_passed:
        print("✅ Database state properly reflects checkpoints for resumability")
    else:
        print("❌ Database state verification FAILED")
    
    # Final summary
    total_passed = sum(1 for r in results if r['success'])
    total_suites = len(results)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Test Suites: {total_passed}/{total_suites} passed")
    print(f"Database State Check: {'PASSED' if db_check_passed else 'FAILED'}")
    
    if total_passed == total_suites and db_check_passed:
        print("\n🎉 ALL TESTS PASSED! HITL flow is working correctly.")
        print("✅ Session resumability is functional")
        print("✅ Database persistence is working")
        print("✅ Attachment handling is implemented")
        print("✅ Edge cases are handled properly")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED! Please review the failures above.")
        if not db_check_passed:
            print("❌ Database state does not properly reflect resumability requirements")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
