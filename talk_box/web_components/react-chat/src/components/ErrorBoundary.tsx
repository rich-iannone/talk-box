import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends React.Component<
  React.PropsWithChildren<{}>,
  ErrorBoundaryState
> {
  constructor(props: React.PropsWithChildren<{}>) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '2rem',
          backgroundColor: '#fee2e2',
          border: '1px solid #fca5a5',
          borderRadius: '0.5rem',
          margin: '1rem'
        }}>
          <h2 style={{ color: '#dc2626', marginBottom: '1rem' }}>Something went wrong</h2>
          <div style={{
            backgroundColor: '#fef2f2',
            padding: '1rem',
            borderRadius: '0.25rem',
            marginBottom: '1rem',
            fontSize: '0.875rem',
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            overflowX: 'auto'
          }}>
            <strong>Error:</strong> {this.state.error?.message || 'Unknown error'}
            {this.state.error?.stack && (
              <>
                <br /><br />
                <strong>Stack Trace:</strong>
                <br />
                {this.state.error.stack}
              </>
            )}
          </div>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: undefined });
              window.location.reload(); // Force full reload
            }}
            style={{
              marginRight: '0.5rem',
              padding: '0.5rem 1rem',
              backgroundColor: '#dc2626',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer'
            }}
          >
            Reload Page
          </button>
          <button
            onClick={() => this.setState({ hasError: false, error: undefined })}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#6b7280',
              color: 'white',
              border: 'none',
              borderRadius: '0.25rem',
              cursor: 'pointer'
            }}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
