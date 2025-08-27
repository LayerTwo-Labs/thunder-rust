//! # CUDA Ed25519 Signature Verification Library
//!
//! This library provides simple threaded CUDA-accelerated verification of Ed25519 signatures,
//! designed for straightforward GPU parallelization where each CUDA thread handles one signature.
//!
//! ## Key Features
//! - **Simple Threading**: Each GPU thread verifies one signature independently
//! - **Easy to Understand**: Straightforward approach without complex batch optimizations
//! - **CUDA Acceleration**: Leverages GPU parallel processing for performance
//! - **Memory Efficient**: Optimized GPU memory management and transfer
//! - **Device Management**: Automatic CUDA device detection and initialization
//!
//! ## Architecture
//!
//! This crate implements a simple threading model:
//! 1. **Data Preparation**: Flatten signature arrays for efficient GPU transfer
//! 2. **Memory Transfer**: Copy data from host to GPU memory
//! 3. **Kernel Launch**: Launch CUDA threads (one per signature)
//! 4. **Individual Verification**: Each thread verifies its assigned signature
//! 5. **Result Collection**: Copy verification results back to host
//!
//! This approach is easier to understand and debug compared to complex batch optimization
//! techniques, while still providing significant performance benefits from GPU parallelization.
//!
//! ## Usage Examples
//!
//! ### Basic Usage
//! ```no_run
//! use cuda_ed25519_verify::{CudaEd25519Verifier, test_data::generate_test_data};
//!
//! // Generate test signatures
//! let (signatures, messages, public_keys) = generate_test_data(1000)?;
//!
//! // Create verifier
//! let mut verifier = CudaEd25519Verifier::new()?;
//!
//! // Verify batch using simple GPU threading
//! let (results, sigs_per_sec) = verifier.verify_batch(&signatures, &messages, &public_keys)?;
//!
//! // Check results
//! let valid_count = results.iter().filter(|&&valid| valid).count();
//! println!("Verified {}/{} signatures at {:.0} sigs/sec",
//!          valid_count, results.len(), sigs_per_sec);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Device Selection
//! ```no_run
//! use cuda_ed25519_verify::CudaEd25519Verifier;
//!
//! // Use specific GPU device
//! let mut verifier = CudaEd25519Verifier::with_device(1)?;
//!
//! // Get device information
//! let device_info = verifier.device_info()?;
//! println!("Using device: {} (Compute {}.{})",
//!          device_info.name,
//!          device_info.compute_capability_major,
//!          device_info.compute_capability_minor);
//!
//! // Get optimization recommendations
//! let recommended_batch_size = device_info.recommended_batch_size();
//! println!("Recommended batch size: {}", recommended_batch_size);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Single Signature Verification
//! ```no_run
//! use cuda_ed25519_verify::{CudaEd25519Verifier, test_data::generate_test_data};
//!
//! let (signatures, messages, public_keys) = generate_test_data(1)?;
//! let mut verifier = CudaEd25519Verifier::new()?;
//!
//! // Verify single signature
//! let is_valid = verifier.verify_single(&signatures[0], &messages[0], &public_keys[0])?;
//! println!("Signature is {}", if is_valid { "valid" } else { "invalid" });
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#[cfg(feature = "cuda")]
pub mod cuda_bindings;
pub mod error;
#[cfg(feature = "cuda")]
pub mod gpu_verifier;
pub mod test_data;

pub use error::{CudaError, Result};
#[cfg(feature = "cuda")]
pub use gpu_verifier::{CudaEd25519Verifier, DeviceInfo};

// Re-export commonly used types
pub use ed25519_dalek::{Signature, VerifyingKey};
pub use test_data::{
    generate_test_data, Hash, OutPoint, Output, Proof, Transaction,
};

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get information about available CUDA devices
#[cfg(feature = "cuda")]
pub fn cuda_device_info() -> Result<CudaDeviceInfo> {
    let device_count = cuda_bindings::get_cuda_device_count()?;

    Ok(CudaDeviceInfo {
        device_count,
        default_device: 0,
    })
}

/// Information about CUDA environment
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Number of CUDA devices available
    pub device_count: i32,
    /// Default device ID to use
    pub default_device: i32,
}

#[cfg(feature = "cuda")]
impl CudaDeviceInfo {
    /// Check if CUDA is available
    pub fn is_cuda_available(&self) -> bool {
        self.device_count > 0
    }

    /// Get list of available device IDs
    pub fn available_devices(&self) -> Vec<i32> {
        (0..self.device_count).collect()
    }

    /// Get recommended device for general use
    pub fn recommended_device(&self) -> Option<i32> {
        if self.device_count > 0 {
            Some(self.default_device)
        } else {
            None
        }
    }
}

#[cfg(not(feature = "cuda"))]
/// Get information about available CUDA devices (CUDA feature disabled)
pub fn cuda_device_info() -> Result<()> {
    Err(CudaError::General(
        "CUDA feature is not enabled in this build".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_version() {
        assert!(!VERSION.is_empty());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_device_detection() {
        let device_info = cuda_device_info().unwrap();
        assert!(device_info.device_count >= 0);

        if device_info.is_cuda_available() {
            let devices = device_info.available_devices();
            assert!(!devices.is_empty());
            assert!(device_info.recommended_device().is_some());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_basic_verification_flow() {
        let (signatures, messages, public_keys) =
            test_data::generate_test_data(10).unwrap();

        let mut verifier = CudaEd25519Verifier::new().unwrap();
        let (results, sigs_per_sec) = verifier
            .verify_batch(&signatures, &messages, &public_keys)
            .unwrap();

        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|&valid| valid));
        assert!(sigs_per_sec > 0.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_single_signature_verification() {
        let (signatures, messages, public_keys) =
            test_data::generate_test_data(1).unwrap();

        let mut verifier = CudaEd25519Verifier::new().unwrap();
        let result = verifier
            .verify_single(&signatures[0], &messages[0], &public_keys[0])
            .unwrap();

        assert!(result);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_device_info() {
        let verifier = CudaEd25519Verifier::new().unwrap();
        let device_info = verifier.device_info().unwrap();

        assert!(device_info.is_compatible());
        assert!(device_info.recommended_batch_size() > 0);
        assert!(device_info.optimal_threads_per_block() > 0);
    }

    #[test]
    fn test_empty_batch() {
        // This test doesn't require CUDA hardware
        let signatures: Vec<[u8; 64]> = vec![];
        let messages: Vec<Vec<u8>> = vec![];
        let public_keys: Vec<[u8; 32]> = vec![];

        #[cfg(feature = "cuda")]
        {
            if let Ok(mut verifier) = CudaEd25519Verifier::new() {
                let (results, _) = verifier
                    .verify_batch(&signatures, &messages, &public_keys)
                    .unwrap();
                assert!(results.is_empty());
            }
        }
    }

    #[test]
    fn test_input_validation() {
        let signatures = vec![[0u8; 64]; 3];
        let messages = vec![vec![0u8; 10]; 2]; // Wrong length
        let public_keys = vec![[0u8; 32]; 3];

        #[cfg(feature = "cuda")]
        {
            if let Ok(mut verifier) = CudaEd25519Verifier::new() {
                let result =
                    verifier.verify_batch(&signatures, &messages, &public_keys);
                assert!(result.is_err());

                if let Err(CudaError::InvalidInput(msg)) = result {
                    assert!(msg.contains("length mismatch"));
                } else {
                    panic!("Expected InvalidInput error");
                }
            }
        }
    }
}
