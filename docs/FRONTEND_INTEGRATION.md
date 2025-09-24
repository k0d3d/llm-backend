# HITL Frontend Integration Guide

This guide provides comprehensive instructions for integrating the Human-in-the-Loop (HITL) system with React-based frontend applications.

## Table of Contents

1. [Overview](#overview)
2. [API Integration](#api-integration)
3. [WebSocket Integration](#websocket-integration)
4. [React Components](#react-components)
5. [State Management](#state-management)
6. [Error Handling](#error-handling)
7. [User Experience Patterns](#user-experience-patterns)
8. [Complete Examples](#complete-examples)

## Overview

The HITL system provides enhanced AI workflow control with human oversight through:
- **Pre-execution validation checkpoints**
- **Real-time WebSocket communication**
- **Interactive approval workflows**
- **Comprehensive error handling**
- **Parameter validation and guidance**

## API Integration

### Basic HITL Endpoint Usage

```typescript
// Types for HITL API responses
interface HITLRunResponse {
  run_id: string;
  status: 'queued' | 'running' | 'awaiting_human' | 'completed' | 'failed' | 'cancelled';
  message?: string;
  websocket_url?: string;
  hitl_enabled: boolean;
  current_step?: string;
  actions_required?: string[];
  approval_token?: string;
  expires_at?: string;
  events_url?: string;
}

interface ValidationCheckpoint {
  type: string;
  title: string;
  description: string;
  required: boolean;
  passed: boolean;
  blocking: boolean;
  user_input_required: boolean;
  issues: ValidationIssue[];
}

interface ValidationIssue {
  field: string;
  issue: string;
  severity: 'error' | 'warning';
  suggested_fix: string;
  auto_fixable: boolean;
}

// Start HITL-enabled run
async function startHITLRun(runInput: RunInput): Promise<HITLRunResponse> {
  const response = await fetch('/api/teams/run?enable_hitl=true', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(runInput)
  });
  
  if (!response.ok) {
    throw new Error(`HITL run failed: ${response.statusText}`);
  }
  
  return response.json();
}

// Get run status with validation details
async function getRunStatus(runId: string): Promise<HITLRunResponse> {
  const response = await fetch(`/api/hitl/runs/${runId}/status`);
  return response.json();
}

// Submit approval for pending checkpoint
async function submitApproval(
  runId: string, 
  approvalToken: string, 
  approved: boolean,
  modifications?: Record<string, any>
): Promise<void> {
  await fetch(`/api/hitl/runs/${runId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      approval_token: approvalToken,
      approved,
      modifications
    })
  });
}
```

### Enhanced Run Input with HITL Support

```typescript
interface EnhancedRunInput {
  prompt: string;
  document_url?: string;
  agent_tool_config: {
    REPLICATETOOL: {
      data: {
        model_name: string;
        description: string;
        example_input: Record<string, any>;
        latest_version: string;
      }
    }
  };
  // HITL-specific options
  hitl_config?: {
    require_approval: boolean;
    policy: 'auto' | 'require_human' | 'auto_with_thresholds';
    allowed_steps: string[];
    review_thresholds?: Record<string, number>;
  };
}
```

## WebSocket Integration

### WebSocket Hook for HITL Communication

```typescript
import { useEffect, useState, useCallback } from 'react';
import useWebSocket from 'react-use-websocket';

interface HITLWebSocketMessage {
  type: 'hitl_approval_request' | 'hitl_status_update' | 'hitl_error' | 'hitl_completion';
  run_id: string;
  data: any;
}

export function useHITLWebSocket(sessionId: string, onMessage?: (message: HITLWebSocketMessage) => void) {
  const [lastHITLMessage, setLastHITLMessage] = useState<HITLWebSocketMessage | null>(null);
  
  const { lastMessage, sendMessage } = useWebSocket(
    `wss://ws.tohju.com`,
    {
      onOpen: () => {
        // Subscribe to HITL events for this session
        sendMessage(JSON.stringify({
          type: 'subscribe',
          channel: `hitl_${sessionId}`
        }));
      },
      onMessage: (event) => {
        try {
          const message = JSON.parse(event.data);
          if (message.type?.startsWith('hitl_')) {
            setLastHITLMessage(message);
            onMessage?.(message);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      },
      shouldReconnect: () => true,
    }
  );

  const sendHITLMessage = useCallback((message: any) => {
    sendMessage(JSON.stringify({
      type: 'hitl_message',
      session_id: sessionId,
      ...message
    }));
  }, [sendMessage, sessionId]);

  return {
    lastHITLMessage,
    sendHITLMessage
  };
}
```

## React Components

### Validation Checkpoint Display Component

```tsx
import React from 'react';
import { AlertTriangle, CheckCircle, XCircle, Upload, Edit } from 'lucide-react';

interface ValidationCheckpointProps {
  checkpoint: ValidationCheckpoint;
  onFixIssue?: (field: string, value: any) => void;
}

export function ValidationCheckpointCard({ checkpoint, onFixIssue }: ValidationCheckpointProps) {
  const getStatusIcon = () => {
    if (checkpoint.passed) return <CheckCircle className="w-5 h-5 text-green-500" />;
    if (checkpoint.blocking) return <XCircle className="w-5 h-5 text-red-500" />;
    return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
  };

  const getStatusColor = () => {
    if (checkpoint.passed) return 'border-green-200 bg-green-50';
    if (checkpoint.blocking) return 'border-red-200 bg-red-50';
    return 'border-yellow-200 bg-yellow-50';
  };

  return (
    <div className={`border rounded-lg p-4 ${getStatusColor()}`}>
      <div className="flex items-center gap-3 mb-3">
        {getStatusIcon()}
        <div>
          <h3 className="font-semibold text-gray-900">{checkpoint.title}</h3>
          <p className="text-sm text-gray-600">{checkpoint.description}</p>
        </div>
      </div>

      {checkpoint.issues.length > 0 && (
        <div className="space-y-2">
          {checkpoint.issues.map((issue, index) => (
            <ValidationIssueItem
              key={index}
              issue={issue}
              onFix={onFixIssue}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ValidationIssueItem({ 
  issue, 
  onFix 
}: { 
  issue: ValidationIssue; 
  onFix?: (field: string, value: any) => void;
}) {
  const getSeverityColor = () => {
    return issue.severity === 'error' ? 'text-red-600' : 'text-yellow-600';
  };

  const handleQuickFix = () => {
    if (issue.field === 'prompt' || issue.field.includes('text')) {
      const newValue = prompt('Enter the required text:');
      if (newValue) onFix?.(issue.field, newValue);
    } else if (issue.field.includes('image') || issue.field.includes('file')) {
      // Trigger file upload
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = issue.field.includes('image') ? 'image/*' : '*/*';
      input.onchange = (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) onFix?.(issue.field, file);
      };
      input.click();
    }
  };

  return (
    <div className="flex items-start gap-3 p-3 bg-white rounded border">
      <div className="flex-1">
        <p className={`font-medium ${getSeverityColor()}`}>
          {issue.issue}
        </p>
        <p className="text-sm text-gray-600 mt-1">
          {issue.suggested_fix}
        </p>
      </div>
      
      {!issue.auto_fixable && onFix && (
        <button
          onClick={handleQuickFix}
          className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
        >
          {issue.field.includes('file') || issue.field.includes('image') ? (
            <Upload className="w-4 h-4" />
          ) : (
            <Edit className="w-4 h-4" />
          )}
          Fix
        </button>
      )}
    </div>
  );
}
```

### HITL Approval Interface Component

```tsx
import React, { useState } from 'react';
import { Play, Pause, Edit, X } from 'lucide-react';

interface HITLApprovalProps {
  runId: string;
  approvalToken: string;
  message: string;
  actionsRequired: string[];
  validationSummary?: {
    checkpoints: ValidationCheckpoint[];
    blocking_issues: number;
    ready_for_execution: boolean;
  };
  onApproval: (approved: boolean, modifications?: Record<string, any>) => void;
}

export function HITLApprovalInterface({
  runId,
  approvalToken,
  message,
  actionsRequired,
  validationSummary,
  onApproval
}: HITLApprovalProps) {
  const [showDetails, setShowDetails] = useState(false);
  const [modifications, setModifications] = useState<Record<string, any>>({});

  const handleApprove = () => {
    onApproval(true, Object.keys(modifications).length > 0 ? modifications : undefined);
  };

  const handleReject = () => {
    onApproval(false);
  };

  const handleFixIssue = (field: string, value: any) => {
    setModifications(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-3">
          <Pause className="w-5 h-5 text-yellow-500" />
          <div>
            <h3 className="font-semibold text-gray-900">Human Approval Required</h3>
            <p className="text-sm text-gray-600">{message}</p>
          </div>
        </div>
      </div>

      {/* Validation Summary */}
      {validationSummary && (
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium text-gray-900">Validation Results</h4>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              {showDetails ? 'Hide Details' : 'Show Details'}
            </button>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {validationSummary.checkpoints.filter(cp => cp.passed).length}
              </div>
              <div className="text-gray-600">Passed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {validationSummary.blocking_issues}
              </div>
              <div className="text-gray-600">Blocking Issues</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${validationSummary.ready_for_execution ? 'text-green-600' : 'text-red-600'}`}>
                {validationSummary.ready_for_execution ? 'Ready' : 'Not Ready'}
              </div>
              <div className="text-gray-600">For Execution</div>
            </div>
          </div>

          {showDetails && (
            <div className="mt-4 space-y-3">
              {validationSummary.checkpoints.map((checkpoint, index) => (
                <ValidationCheckpointCard
                  key={index}
                  checkpoint={checkpoint}
                  onFixIssue={handleFixIssue}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Required Actions */}
      <div className="p-4 border-b border-gray-200">
        <h4 className="font-medium text-gray-900 mb-2">Required Actions</h4>
        <ul className="space-y-1">
          {actionsRequired.map((action, index) => (
            <li key={index} className="flex items-center gap-2 text-sm text-gray-600">
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full" />
              {action.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </li>
          ))}
        </ul>
      </div>

      {/* Modifications Preview */}
      {Object.keys(modifications).length > 0 && (
        <div className="p-4 border-b border-gray-200 bg-blue-50">
          <h4 className="font-medium text-gray-900 mb-2">Pending Modifications</h4>
          <div className="space-y-1">
            {Object.entries(modifications).map(([field, value]) => (
              <div key={field} className="text-sm">
                <span className="font-medium">{field}:</span>{' '}
                <span className="text-gray-600">
                  {typeof value === 'string' ? value : JSON.stringify(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="p-4 flex gap-3">
        <button
          onClick={handleApprove}
          disabled={validationSummary && !validationSummary.ready_for_execution}
          className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          <Play className="w-4 h-4" />
          Approve & Continue
        </button>
        
        <button
          onClick={handleReject}
          className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          <X className="w-4 h-4" />
          Cancel
        </button>
        
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50"
        >
          <Edit className="w-4 h-4" />
          Edit Parameters
        </button>
      </div>
    </div>
  );
}
```

## State Management

### HITL State Management Hook

```tsx
import { useState, useEffect, useCallback } from 'react';

interface HITLState {
  runId: string | null;
  status: string;
  currentStep: string | null;
  validationSummary: any | null;
  approvalPending: boolean;
  approvalToken: string | null;
  error: string | null;
}

export function useHITLState(sessionId: string) {
  const [state, setState] = useState<HITLState>({
    runId: null,
    status: 'idle',
    currentStep: null,
    validationSummary: null,
    approvalPending: false,
    approvalToken: null,
    error: null
  });

  const { lastHITLMessage, sendHITLMessage } = useHITLWebSocket(
    sessionId,
    useCallback((message) => {
      switch (message.type) {
        case 'hitl_approval_request':
          setState(prev => ({
            ...prev,
            status: 'awaiting_human',
            approvalPending: true,
            approvalToken: message.data.approval_token,
            validationSummary: message.data.validation_summary,
            currentStep: message.data.current_step
          }));
          break;

        case 'hitl_status_update':
          setState(prev => ({
            ...prev,
            status: message.data.status,
            currentStep: message.data.current_step,
            approvalPending: false
          }));
          break;

        case 'hitl_error':
          setState(prev => ({
            ...prev,
            status: 'error',
            error: message.data.error,
            approvalPending: false
          }));
          break;

        case 'hitl_completion':
          setState(prev => ({
            ...prev,
            status: 'completed',
            approvalPending: false,
            currentStep: 'completed'
          }));
          break;
      }
    }, [])
  );

  const startHITLRun = useCallback(async (runInput: EnhancedRunInput) => {
    try {
      setState(prev => ({ ...prev, error: null }));
      const response = await startHITLRun(runInput);
      
      setState(prev => ({
        ...prev,
        runId: response.run_id,
        status: response.status
      }));

      return response;
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 'error'
      }));
      throw error;
    }
  }, []);

  const submitApproval = useCallback(async (
    approved: boolean,
    modifications?: Record<string, any>
  ) => {
    if (!state.runId || !state.approvalToken) return;

    try {
      await submitApproval(state.runId, state.approvalToken, approved, modifications);
      
      setState(prev => ({
        ...prev,
        approvalPending: false,
        approvalToken: null
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Approval failed'
      }));
    }
  }, [state.runId, state.approvalToken]);

  const resetState = useCallback(() => {
    setState({
      runId: null,
      status: 'idle',
      currentStep: null,
      validationSummary: null,
      approvalPending: false,
      approvalToken: null,
      error: null
    });
  }, []);

  return {
    state,
    startHITLRun,
    submitApproval,
    resetState
  };
}
```

## Error Handling

### Error Boundary for HITL Components

```tsx
import React, { Component, ReactNode } from 'react';

interface HITLErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class HITLErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  HITLErrorBoundaryState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): HITLErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('HITL Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-4 border border-red-200 rounded-lg bg-red-50">
          <h3 className="font-semibold text-red-800 mb-2">HITL System Error</h3>
          <p className="text-red-600 text-sm">
            {this.state.error?.message || 'An unexpected error occurred in the HITL system.'}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-3 px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
```

## User Experience Patterns

### Progressive Enhancement Pattern

```tsx
import React, { useState } from 'react';

interface AIWorkflowProps {
  onSubmit: (input: EnhancedRunInput) => void;
  hitlEnabled?: boolean;
}

export function AIWorkflowForm({ onSubmit, hitlEnabled = false }: AIWorkflowProps) {
  const [input, setInput] = useState<EnhancedRunInput>({
    prompt: '',
    agent_tool_config: {
      REPLICATETOOL: {
        data: {
          model_name: '',
          description: '',
          example_input: {},
          latest_version: ''
        }
      }
    }
  });

  const [validationMode, setValidationMode] = useState<'basic' | 'enhanced'>('basic');

  useEffect(() => {
    if (hitlEnabled) {
      setValidationMode('enhanced');
    }
  }, [hitlEnabled]);

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={validationMode === 'enhanced'}
            onChange={(e) => setValidationMode(e.target.checked ? 'enhanced' : 'basic')}
          />
          <span className="text-sm font-medium">Enhanced validation (HITL)</span>
        </label>
      </div>

      {/* Enhanced mode benefits */}
      {validationMode === 'enhanced' && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-1">Enhanced Mode Benefits</h4>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Pre-execution parameter validation</li>
            <li>• Human approval checkpoints</li>
            <li>• Detailed error guidance</li>
            <li>• Auto-fix suggestions</li>
          </ul>
        </div>
      )}

      {/* Form fields */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Prompt *
          </label>
          <textarea
            value={input.prompt}
            onChange={(e) => setInput(prev => ({ ...prev, prompt: e.target.value }))}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            rows={4}
            placeholder="Describe what you want the AI to do..."
          />
        </div>

        {/* File upload */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Upload File (Optional)
          </label>
          <input
            type="file"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                // Handle file upload
                setInput(prev => ({ ...prev, document_url: URL.createObjectURL(file) }));
              }
            }}
            className="w-full p-2 border border-gray-300 rounded-lg"
          />
        </div>
      </div>

      {/* Submit button */}
      <button
        onClick={() => onSubmit({
          ...input,
          hitl_config: validationMode === 'enhanced' ? {
            require_approval: true,
            policy: 'auto_with_thresholds',
            allowed_steps: ['information_review', 'payload_review', 'response_review']
          } : undefined
        })}
        disabled={!input.prompt.trim()}
        className="w-full py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
      >
        {validationMode === 'enhanced' ? 'Start Enhanced AI Workflow' : 'Start AI Workflow'}
      </button>
    </div>
  );
}
```

## Complete Examples

### Full HITL Integration Example

```tsx
import React from 'react';
import { HITLErrorBoundary } from './HITLErrorBoundary';
import { useHITLState } from './useHITLState';
import { HITLApprovalInterface } from './HITLApprovalInterface';
import { AIWorkflowForm } from './AIWorkflowForm';

export function HITLWorkflowContainer({ sessionId }: { sessionId: string }) {
  const { state, startHITLRun, submitApproval, resetState } = useHITLState(sessionId);

  const handleSubmit = async (input: EnhancedRunInput) => {
    try {
      await startHITLRun(input);
    } catch (error) {
      console.error('Failed to start HITL run:', error);
    }
  };

  const handleApproval = async (approved: boolean, modifications?: Record<string, any>) => {
    await submitApproval(approved, modifications);
  };

  return (
    <HITLErrorBoundary>
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <h1 className="text-2xl font-bold text-gray-900">AI Workflow with HITL</h1>

        {/* Status Display */}
        {state.status !== 'idle' && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-gray-600">Status:</span>
                <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${
                  state.status === 'completed' ? 'bg-green-100 text-green-800' :
                  state.status === 'error' ? 'bg-red-100 text-red-800' :
                  state.status === 'awaiting_human' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-blue-100 text-blue-800'
                }`}>
                  {state.status.replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <button
                onClick={resetState}
                className="text-sm text-gray-600 hover:text-gray-800"
              >
                Reset
              </button>
            </div>
            
            {state.currentStep && (
              <div className="mt-2">
                <span className="text-sm text-gray-600">Current Step: {state.currentStep}</span>
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {state.error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <h3 className="font-medium text-red-800">Error</h3>
            <p className="text-red-600 text-sm mt-1">{state.error}</p>
          </div>
        )}

        {/* Approval Interface */}
        {state.approvalPending && state.approvalToken && (
          <HITLApprovalInterface
            runId={state.runId!}
            approvalToken={state.approvalToken}
            message="Please review the validation results and approve to continue."
            actionsRequired={['review_validation', 'approve_execution']}
            validationSummary={state.validationSummary}
            onApproval={handleApproval}
          />
        )}

        {/* Workflow Form */}
        {state.status === 'idle' && (
          <AIWorkflowForm
            onSubmit={handleSubmit}
            hitlEnabled={true}
          />
        )}

        {/* Results Display */}
        {state.status === 'completed' && (
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <h3 className="font-medium text-green-800">Workflow Completed Successfully</h3>
            <p className="text-green-600 text-sm mt-1">
              Your AI workflow has been completed with human oversight.
            </p>
          </div>
        )}
      </div>
    </HITLErrorBoundary>
  );
}
```

## Best Practices

### 1. Progressive Enhancement
- Start with basic AI workflows
- Add HITL features as optional enhancements
- Provide clear benefits explanation

### 2. User Feedback
- Show validation progress clearly
- Provide actionable error messages
- Guide users through fixing issues

### 3. Performance
- Use WebSocket connections efficiently
- Cache validation results when possible
- Implement proper loading states

### 4. Accessibility
- Ensure keyboard navigation works
- Provide screen reader support
- Use semantic HTML elements

### 5. Error Recovery
- Implement retry mechanisms
- Provide fallback options
- Save user progress when possible

This integration guide provides everything needed to implement HITL functionality in React applications, from basic API integration to complete user interfaces with real-time WebSocket communication.
