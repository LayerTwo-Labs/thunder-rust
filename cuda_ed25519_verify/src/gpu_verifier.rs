//! Simple threaded GPU verifier for Ed25519 signature verification
//!
//! This module provides a straightforward GPU-based verifier where each CUDA thread
//! verifies one signature independently. This is the simple approach without complex
//! batch optimization techniques like MSM.

use crate::cuda_bindings::{get_cuda_device_count, verify_signatures_cuda};
use crate::error::{CudaError, Result};
use std::sync::Arc;
use std::sync::Mutex;

/// Simple CUDA-threaded Ed25519 signature verifier
///
/// This verifier launches individual GPU threads for signature verification.
/// Each thread handles one signature, making it straightforward and easy to understand.
///
/// # Examples
///
/// ```no_run
/// use cuda_ed25519_verify::{CudaEd25519Verifier, test_data::generate_test_data};
///
/// // Generate test data
/// let (signatures, messages, public_keys) = generate_test_data(1000)?;
///
/// // Create verifier and verify batch
/// let mut verifier = CudaEd25519Verifier::new()?;
/// let (results, perf) = verifier.verify_batch(&signatures, &messages, &public_keys)?;
///
/// // Check results
/// let valid_count = results.iter().filter(|&&valid| valid).count();
/// println!("Verified {}/{} signatures at {:.0} sigs/sec", valid_count, results.len(), perf);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CudaEd25519Verifier {
    device_id: i32,
    _context: Arc<Mutex<()>>, // CUDA context protection
}

impl CudaEd25519Verifier {
    /// Create a new CUDA Ed25519 verifier
    ///
    /// This initializes the CUDA runtime and selects the first available GPU device.
    /// Returns an error if no compatible CUDA device is found.
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Create a new verifier using a specific CUDA device
    ///
    /// # Arguments
    /// * `device_id` - The CUDA device ID to use (typically 0 for the first GPU)
    pub fn with_device(device_id: i32) -> Result<Self> {
        // Initialize CUDA and check device availability
        Self::check_cuda_device(device_id)?;

        Ok(CudaEd25519Verifier {
            device_id,
            _context: Arc::new(Mutex::new(())),
        })
    }

    /// Verify a batch of Ed25519 signatures using simple GPU threading
    ///
    /// This launches individual GPU threads for signature verification - one thread per signature.
    /// This is the straightforward approach that's easy to understand and debug.
    ///
    /// # Arguments
    /// * `signatures` - Array of 64-byte Ed25519 signatures to verify
    /// * `messages` - Array of messages that were signed (variable length)
    /// * `public_keys` - Array of 32-byte public keys corresponding to each signature
    ///
    /// # Returns
    /// A tuple containing:
    /// - Vector of boolean values indicating whether each signature is valid
    /// - Signatures per second performance metric
    ///
    /// # Errors
    /// Returns `CudaError` if:
    /// - Input arrays have mismatched lengths
    /// - GPU memory allocation fails
    /// - CUDA kernel execution fails
    pub fn verify_batch(
        &mut self,
        signatures: &[[u8; 64]],
        messages: &[Vec<u8>],
        public_keys: &[[u8; 32]],
    ) -> Result<(Vec<bool>, f64)> {
        if signatures.is_empty() {
            return Ok((Vec::new(), 0.0));
        }

        // Validate input consistency
        if signatures.len() != messages.len()
            || signatures.len() != public_keys.len()
        {
            return Err(CudaError::InvalidInput(
                format!(
                    "Input array length mismatch: signatures={}, messages={}, public_keys={}",
                    signatures.len(), messages.len(), public_keys.len()
                )
            ));
        }

        // Use GPU acceleration with simple threading approach
        let _lock = self._context.lock().map_err(|_| {
            CudaError::General(
                "Failed to acquire CUDA context lock".to_string(),
            )
        })?;

        // Launch individual GPU threads for verification
        verify_signatures_cuda(signatures, messages, public_keys)
    }

    /// Verify a single signature (convenience method)
    ///
    /// This is a convenience wrapper around `verify_batch` for single signature verification.
    /// For better GPU utilization with multiple signatures, use `verify_batch` directly.
    pub fn verify_single(
        &mut self,
        signature: &[u8; 64],
        message: &[u8],
        public_key: &[u8; 32],
    ) -> Result<bool> {
        let (results, _perf) = self.verify_batch(
            &[*signature],
            &[message.to_vec()],
            &[*public_key],
        )?;

        Ok(results[0])
    }

    /// Get information about the CUDA device being used
    pub fn device_info(&self) -> Result<DeviceInfo> {
        Self::get_device_info(self.device_id)
    }

    /// Check CUDA device availability and compatibility
    fn check_cuda_device(device_id: i32) -> Result<()> {
        if device_id < 0 {
            return Err(CudaError::InvalidInput(
                "Device ID must be non-negative".to_string(),
            ));
        }

        // Check if CUDA devices are available
        let device_count = get_cuda_device_count()?;
        if device_count == 0 {
            return Err(CudaError::General(
                "No CUDA devices available".to_string(),
            ));
        }

        if device_id >= device_count {
            return Err(CudaError::InvalidInput(format!(
                "Device ID {} exceeds available devices ({})",
                device_id, device_count
            )));
        }

        Ok(())
    }

    /// Get detailed information about a CUDA device
    fn get_device_info(device_id: i32) -> Result<DeviceInfo> {
        // TODO: Implement actual device property queries
        // For now, return placeholder information
        Ok(DeviceInfo {
            device_id,
            name: "CUDA Device".to_string(),
            compute_capability_major: 7,
            compute_capability_minor: 5,
            total_global_memory: 8 * 1024 * 1024 * 1024, // 8GB placeholder
            max_threads_per_block: 1024,
            max_blocks_per_grid: 65535,
        })
    }
}

/// Information about a CUDA device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_global_memory: usize,
    pub max_threads_per_block: i32,
    pub max_blocks_per_grid: i32,
}

impl DeviceInfo {
    /// Check if the device supports the required compute capability for Ed25519
    pub fn is_compatible(&self) -> bool {
        // Ed25519 CUDA implementation requires at least compute capability 5.0
        self.compute_capability_major >= 5
    }

    /// Get recommended batch size for optimal GPU utilization
    pub fn recommended_batch_size(&self) -> usize {
        // Base recommendation on available memory and compute capability
        let base_size = match self.compute_capability_major {
            5 => 1024,  // Maxwell
            6 => 2048,  // Pascal
            7 => 4096,  // Volta/Turing
            8 => 8192,  // Ampere
            9 => 16384, // Ada Lovelace
            _ => 1024,  // Conservative default
        };

        // Adjust based on available memory (rough estimate)
        let memory_factor =
            (self.total_global_memory / (1024 * 1024 * 1024)).min(8); // Cap at 8GB consideration
        base_size * memory_factor.max(1)
    }

    /// Get optimal thread block size for this device
    pub fn optimal_threads_per_block(&self) -> i32 {
        // Use device-specific optimization
        match self.compute_capability_major {
            5 => 128, // Maxwell: smaller blocks often better
            6 => 256, // Pascal: balanced approach
            7 => 256, // Volta/Turing: 256 is usually optimal
            8 => 512, // Ampere: can handle larger blocks efficiently
            9 => 512, // Ada Lovelace: similar to Ampere
            _ => 256, // Safe default
        }
    }
}

// Ensure thread safety
unsafe impl Send for CudaEd25519Verifier {}
unsafe impl Sync for CudaEd25519Verifier {}

impl Default for CudaEd25519Verifier {
    fn default() -> Self {
        Self::new().expect("Failed to create default CUDA verifier")
    }
}
