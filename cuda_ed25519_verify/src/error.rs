//! Error handling for CUDA ED25519 operations

use std::fmt;

/// Result type for CUDA operations
pub type Result<T> = std::result::Result<T, CudaError>;

/// Errors that can occur during CUDA ED25519 operations
#[derive(Debug, Clone)]
pub enum CudaError {
    /// CUDA runtime error
    CudaRuntime(String),
    /// Memory allocation error
    MemoryAllocation(String),
    /// Invalid input parameters
    InvalidInput(String),
    /// GPU device not available
    DeviceNotAvailable,
    /// Kernel execution failed
    KernelExecution(String),
    /// Data serialization error
    Serialization(String),
    /// General operation error
    General(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::CudaRuntime(msg) => {
                write!(f, "CUDA runtime error: {}", msg)
            }
            CudaError::MemoryAllocation(msg) => {
                write!(f, "Memory allocation error: {}", msg)
            }
            CudaError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CudaError::DeviceNotAvailable => {
                write!(f, "CUDA device not available")
            }
            CudaError::KernelExecution(msg) => {
                write!(f, "Kernel execution failed: {}", msg)
            }
            CudaError::Serialization(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            CudaError::General(msg) => write!(f, "General error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<anyhow::Error> for CudaError {
    fn from(err: anyhow::Error) -> Self {
        CudaError::General(err.to_string())
    }
}

impl From<borsh::io::Error> for CudaError {
    fn from(err: borsh::io::Error) -> Self {
        CudaError::Serialization(err.to_string())
    }
}
