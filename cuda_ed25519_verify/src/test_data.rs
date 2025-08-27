//! Test data generation module
//!
//! Extracted from the WORKING test_phase1_complete example to ensure consistency
//! with Thunder-rust transaction format and ed25519_dalek signature generation.

use crate::error::Result;
use borsh::{BorshDeserialize, BorshSerialize};
use ed25519_dalek::{Signer, SigningKey};

// Transaction data structures that match Thunder-rust format EXACTLY
// These are copied directly from the working example
#[derive(BorshSerialize, BorshDeserialize, Clone, Debug)]
pub struct Transaction {
    pub inputs: Vec<(OutPoint, Hash)>,
    pub proof: Proof,
    pub outputs: Vec<Output>,
}

#[derive(BorshSerialize, BorshDeserialize, Clone, Debug)]
pub struct OutPoint {
    pub txid: [u8; 32],
    pub vout: u32,
}

// Hash type (32-byte Blake3 hash like Thunder-rust)
pub type Hash = [u8; 32];

// Type alias for test data return type
pub type TestData = (Vec<[u8; 64]>, Vec<Vec<u8>>, Vec<[u8; 32]>);

#[derive(BorshSerialize, BorshDeserialize, Clone, Debug, Default)]
pub struct Proof {
    // Empty proof for benchmarks - matches Thunder-rust Utreexo proof structure
}

#[derive(BorshSerialize, BorshDeserialize, Clone, Debug)]
pub struct Output {
    pub value: u64,
    pub script_pubkey: Vec<u8>,
}

/// Generate test data using the EXACT SAME logic as the working test_phase1_complete example
/// This is the gold standard that we know works perfectly
pub fn generate_test_data(
    batch_size: usize,
) -> Result<TestData> {
    use rand::rngs::OsRng;
    use std::time::Instant;

    // ------- helpers -------------------------------------------------------
    /// Produce a random `OutPoint` (txid + vout) deterministically from (`i`,`j`)
    fn random_outpoint(i: usize, j: usize) -> OutPoint {
        let mut txid = [0u8; 32];
        // deterministic but varied so we get reproducible test‑vectors
        for (k, b) in txid.iter_mut().enumerate() {
            *b = ((i * 7 + j * 11 + k) % 256) as u8;
        }
        OutPoint {
            txid,
            vout: (j % 4) as u32,
        }
    }

    /// Produce a random `Output` worth 1 sat × <0;10 000>
    fn random_output(i: usize, j: usize) -> Output {
        Output {
            value: ((i * 17 + j * 13) % 10_000) as u64 + 1,
            script_pubkey: vec![0u8; 25], // 1 P2PKH‑ish dummy script
        }
    }

    // ------- main loop -----------------------------------------------------
    println!("🔧 Generating {batch_size} *transaction* messages…");
    let start = Instant::now();
    let mut signatures = Vec::with_capacity(batch_size);
    let mut messages = Vec::with_capacity(batch_size);
    let mut public_keys = Vec::with_capacity(batch_size);
    let mut _csprng = OsRng;

    for i in 0..batch_size {
        // ---------------- build Transaction -------------------------------
        let n_inputs = 1 + (i % 4); // 1‒4 inputs
        let n_outputs = 1 + ((i + 1) % 3); // 1‒3 outputs
        let inputs: Vec<(OutPoint, Hash)> = (0..n_inputs)
            .map(|j| {
                (
                    random_outpoint(i, j),
                    [((i + j) % 256) as u8; 32], // dummy prev‑output hash
                )
            })
            .collect();
        let outputs: Vec<Output> =
            (0..n_outputs).map(|j| random_output(i, j)).collect();
        let tx = Transaction {
            inputs,
            proof: Proof::default(), // empty Utreexo proof for benchmarks
            outputs,
        };

        // ---------------- serialise & sign --------------------------------
        let tx_bytes = borsh::to_vec(&tx)?; // Borsh serialisation
        let sk_bytes: [u8; 32] = rand::random(); // raw 32‑byte seed
        let signing_key = SigningKey::from_bytes(&sk_bytes);
        let signature = signing_key.sign(&tx_bytes);

        // ---------------- store artefacts ---------------------------------
        signatures.push(signature.to_bytes());
        messages.push(tx_bytes);
        public_keys.push(signing_key.verifying_key().to_bytes());

        // optional progress for huge runs
        if batch_size >= 10_000 && i % (batch_size / 10) == 0 {
            println!(
                "  built {i}/{batch_size} txs ({:.1} %)",
                (i as f64 / batch_size as f64) * 100.0
            );
        }
    }

    let elapsed = start.elapsed();
    println!(
        "✅ Built {batch_size} txs in {:.2}ms ({:.0} tx/s)",
        elapsed.as_millis(),
        batch_size as f64 / elapsed.as_secs_f64()
    );

    Ok((signatures, messages, public_keys))
}

/// Verify test data using CPU ed25519_dalek for comparison/testing
pub fn verify_test_data_cpu(
    signatures: &[[u8; 64]],
    messages: &[Vec<u8>],
    public_keys: &[[u8; 32]],
) -> Result<Vec<bool>> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    if signatures.len() != messages.len()
        || signatures.len() != public_keys.len()
    {
        return Err(crate::error::CudaError::InvalidInput(
            "Mismatched array lengths for verification".to_string(),
        ));
    }

    let batch_size = signatures.len();

    // Convert to ed25519_dalek types
    let dalek_signatures: Vec<Signature> = signatures
        .iter()
        .map(Signature::from_bytes)
        .collect();
    let dalek_public_keys: std::result::Result<Vec<VerifyingKey>, _> =
        public_keys
            .iter()
            .map(VerifyingKey::from_bytes)
            .collect();
    let dalek_public_keys = dalek_public_keys.map_err(|e| {
        crate::error::CudaError::InvalidInput(format!(
            "Invalid public key: {}",
            e
        ))
    })?;
    let message_refs: Vec<&[u8]> =
        messages.iter().map(|m| m.as_slice()).collect();

    // Try batch verification first (fastest for valid signatures)
    let batch_result = ed25519_dalek::verify_batch(
        &message_refs,
        &dalek_signatures,
        &dalek_public_keys,
    );

    if batch_result.is_ok() {
        // All signatures are valid
        Ok(vec![true; batch_size])
    } else {
        // Batch failed, fall back to individual verification to get per-signature results
        let results: Vec<bool> = (0..batch_size)
            .map(|i| {
                dalek_public_keys[i]
                    .verify(message_refs[i], &dalek_signatures[i])
                    .is_ok()
            })
            .collect();

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_verify_small_batch() {
        let batch_size = 10;
        let (signatures, messages, public_keys) =
            generate_test_data(batch_size).unwrap();

        assert_eq!(signatures.len(), batch_size);
        assert_eq!(messages.len(), batch_size);
        assert_eq!(public_keys.len(), batch_size);

        // Verify with CPU implementation
        let cpu_results =
            verify_test_data_cpu(&signatures, &messages, &public_keys).unwrap();

        // All signatures should be valid
        assert!(
            cpu_results.iter().all(|&valid| valid),
            "Not all generated signatures are valid"
        );
    }

    #[test]
    fn test_transaction_serialization() {
        let tx = Transaction {
            inputs: vec![(
                OutPoint {
                    txid: [1; 32],
                    vout: 0,
                },
                [2; 32],
            )],
            proof: Proof::default(),
            outputs: vec![Output {
                value: 1000,
                script_pubkey: vec![0; 25],
            }],
        };

        let serialized = borsh::to_vec(&tx).unwrap();
        let deserialized: Transaction = borsh::from_slice(&serialized).unwrap();

        assert_eq!(tx.inputs.len(), deserialized.inputs.len());
        assert_eq!(tx.outputs.len(), deserialized.outputs.len());
        assert_eq!(tx.outputs[0].value, deserialized.outputs[0].value);
    }
}
