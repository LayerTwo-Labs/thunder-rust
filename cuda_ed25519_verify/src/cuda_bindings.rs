//! Simple threaded CUDA FFI bindings for Ed25519 signature verification
//!
//! This module provides the core CUDA functionality for individual signature verification
//! using GPU threads - each thread verifies one signature independently.

use crate::error::{CudaError, Result};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uchar};
use std::ptr;
use std::time::Instant;

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// CUDA device memory management
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_uchar, size: usize) -> c_int;
    fn cudaMemcpy(
        dst: *mut c_uchar,
        src: *const c_uchar,
        count: usize,
        kind: c_int,
    ) -> c_int;
    fn cudaFree(devPtr: *mut c_uchar) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const c_char;
    fn cudaGetLastError() -> c_int;
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// Check CUDA error code and convert to Result
pub fn cuda_check(error: c_int) -> Result<()> {
    if error != 0 {
        unsafe {
            let error_str = CStr::from_ptr(cudaGetErrorString(error));
            return Err(CudaError::CudaRuntime(
                error_str.to_string_lossy().into_owned(),
            ));
        }
    }
    Ok(())
}

/// Allocate CUDA device memory
pub fn cuda_malloc(size: usize) -> Result<*mut c_uchar> {
    let mut ptr: *mut c_uchar = ptr::null_mut();
    unsafe {
        cuda_check(cudaMalloc(&mut ptr, size))?;
    }
    if ptr.is_null() {
        return Err(CudaError::MemoryAllocation(
            "cudaMalloc returned null pointer".to_string(),
        ));
    }
    Ok(ptr)
}

/// Copy data from host to device
pub unsafe fn cuda_memcpy_h2d(
    dst: *mut c_uchar,
    src: *const c_uchar,
    count: usize,
) -> Result<()> {
    cuda_check(cudaMemcpy(dst, src, count, CUDA_MEMCPY_HOST_TO_DEVICE))
}

/// Copy data from device to host
pub unsafe fn cuda_memcpy_d2h(
    dst: *mut c_uchar,
    src: *const c_uchar,
    count: usize,
) -> Result<()> {
    cuda_check(cudaMemcpy(dst, src, count, CUDA_MEMCPY_DEVICE_TO_HOST))
}

/// Free CUDA device memory
pub unsafe fn cuda_free(ptr: *mut c_uchar) -> Result<()> {
    cuda_check(cudaFree(ptr))
}

/// Synchronize CUDA device
pub fn cuda_synchronize() -> Result<()> {
    unsafe { cuda_check(cudaDeviceSynchronize()) }
}

/// Calculate optimal grid size for CUDA kernel launch
pub fn calculate_grid_size(
    batch_size: usize,
    preferred_threads_per_block: i32,
) -> (i32, i32) {
    let threads_per_block = preferred_threads_per_block.clamp(32, 1024);
    let blocks =
        (batch_size as i32 + threads_per_block - 1) / threads_per_block;
    (blocks, threads_per_block)
}

/// Simple threaded signature verification using individual GPU threads
///
/// This function launches GPU threads where each thread verifies one signature independently.
/// This is the simple, straightforward approach (NOT the complex MSM batch verification).
pub fn verify_signatures_cuda(
    signatures: &[[u8; 64]],
    messages: &[Vec<u8>],
    public_keys: &[[u8; 32]],
) -> Result<(Vec<bool>, f64)> {
    let start_time = Instant::now();
    let batch_size = signatures.len();

    if batch_size == 0 {
        return Ok((Vec::new(), 0.0));
    }

    if batch_size != messages.len() || batch_size != public_keys.len() {
        return Err(CudaError::InvalidInput(
            "Signatures, messages, and public_keys arrays must have the same length".to_string(),
        ));
    }

    // Prepare data for GPU - flatten arrays for efficient transfer
    let mut flat_signatures = Vec::with_capacity(batch_size * 64);
    for sig in signatures {
        flat_signatures.extend_from_slice(sig);
    }

    let mut flat_public_keys = Vec::with_capacity(batch_size * 32);
    for pk in public_keys {
        flat_public_keys.extend_from_slice(pk);
    }

    // Prepare message data with length offsets (messages have variable length)
    let mut flat_messages = Vec::new();
    let mut message_lengths = Vec::with_capacity(batch_size + 1);
    message_lengths.push(0);

    for msg in messages {
        flat_messages.extend_from_slice(msg);
        message_lengths.push(flat_messages.len());
    }

    // Allocate GPU memory
    let d_signatures = cuda_malloc(flat_signatures.len())?;
    let d_public_keys = cuda_malloc(flat_public_keys.len())?;
    let d_messages = cuda_malloc(flat_messages.len())?;
    let d_message_lengths =
        cuda_malloc(message_lengths.len() * std::mem::size_of::<usize>())?;
    let d_verified = cuda_malloc(batch_size * std::mem::size_of::<c_int>())?;

    // Copy data to GPU
    unsafe {
        cuda_memcpy_h2d(
            d_signatures,
            flat_signatures.as_ptr(),
            flat_signatures.len(),
        )?;
        cuda_memcpy_h2d(
            d_public_keys,
            flat_public_keys.as_ptr(),
            flat_public_keys.len(),
        )?;
        cuda_memcpy_h2d(
            d_messages,
            flat_messages.as_ptr(),
            flat_messages.len(),
        )?;
        cuda_memcpy_h2d(
            d_message_lengths,
            message_lengths.as_ptr() as *const c_uchar,
            message_lengths.len() * std::mem::size_of::<usize>(),
        )?;
    }

    // Launch kernel - calculate grid size for optimal GPU utilization
    let (blocks, threads_per_block) = calculate_grid_size(batch_size, 256);

    unsafe {
        launch_verify_kernel(
            d_signatures,
            d_messages,
            d_message_lengths as *mut usize,
            d_public_keys,
            d_verified as *mut c_int,
            ptr::null_mut(), // No key mapping needed for individual verification
            batch_size as c_int,
            blocks,
            threads_per_block,
        );
    }

    // Check for kernel launch errors
    unsafe {
        let error = cudaGetLastError();
        cuda_check(error)?;
    }

    // Synchronize GPU and copy results back
    cuda_synchronize()?;

    let mut h_verified = vec![0i32; batch_size];
    unsafe {
        cuda_memcpy_d2h(
            h_verified.as_mut_ptr() as *mut c_uchar,
            d_verified,
            batch_size * std::mem::size_of::<c_int>(),
        )?;
    }

    // Clean up GPU memory
    unsafe {
        cuda_free(d_signatures)?;
        cuda_free(d_public_keys)?;
        cuda_free(d_messages)?;
        cuda_free(d_message_lengths)?;
        cuda_free(d_verified)?;
    }

    let execution_time = start_time.elapsed();
    let signatures_per_second =
        batch_size as f64 / execution_time.as_secs_f64();

    let verified_results: Vec<bool> =
        h_verified.iter().map(|&v| v != 0).collect();

    Ok((verified_results, signatures_per_second))
}

/// Get the number of CUDA devices available
pub fn get_cuda_device_count() -> Result<i32> {
    let mut count = 0;
    unsafe {
        cuda_check(cudaGetDeviceCount(&mut count))?;
    }
    Ok(count)
}
