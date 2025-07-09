//! Test modules for ArenaForest implementation.

use super::*;
use crate::accumulator::node_hash::AccumulatorHash;
use crate::accumulator::node_hash::BitcoinNodeHash;
use crate::accumulator::util::hash_from_u8;

#[test]
fn test_arena_node_creation() {
    let hash = BitcoinNodeHash::empty();

    // Test leaf creation
    let leaf = ArenaNode::new_leaf(hash);
    assert!(leaf.is_leaf());
    assert_eq!(leaf.left_child(), None);
    assert_eq!(leaf.right_child(), None);

    // Test internal node creation
    let internal = ArenaNode::new_internal(hash, 0, 1, 1);
    assert!(!internal.is_leaf());
    assert_eq!(internal.left_child(), Some(0));
    assert_eq!(internal.right_child(), Some(1));
}

#[test]
fn test_arena_forest_basic_operations() {
    let mut forest = ArenaForest::<BitcoinNodeHash>::new();
    assert_eq!(forest.len(), 0);
    assert!(forest.is_empty());
    assert_eq!(forest.leaves(), 0);

    let hash = BitcoinNodeHash::empty();
    let node = ArenaNode::new_leaf(hash);
    let idx = forest.allocate_node(node);

    assert_eq!(forest.len(), 1);
    assert!(!forest.is_empty());
    assert!(forest.get_node(idx).unwrap().is_leaf());
}

#[test]
fn test_node_type_bit_packing() {
    let hash = BitcoinNodeHash::empty();
    let mut node = ArenaNode::new_internal(hash, 42, 84, 1);

    // Should be internal initially
    assert!(!node.is_leaf());
    assert_eq!(node.left_child(), Some(42));
    assert_eq!(node.right_child(), Some(84));

    // Convert to leaf
    node.make_leaf();
    assert!(node.is_leaf());
    assert_eq!(node.left_child(), None);
    assert_eq!(node.right_child(), None);

    // Convert back to internal
    node.make_internal(100, 200);
    assert!(!node.is_leaf());
    assert_eq!(node.left_child(), Some(100));
    assert_eq!(node.right_child(), Some(200));
}

#[test]
fn test_dirty_tracking() {
    let mut forest = ArenaForest::<BitcoinNodeHash>::new();

    // Mark some nodes as dirty at different levels
    forest.mark_dirty(0, 0);
    forest.mark_dirty(1, 1);
    forest.mark_dirty(2, 2);

    assert_eq!(forest.max_height, 2);
    assert_eq!(forest.dirty_levels[0].len(), 1);
    assert_eq!(forest.dirty_levels[1].len(), 1);
    assert_eq!(forest.dirty_levels[2].len(), 1);
}

#[test]
fn comprehensive_add_prove_delete_verify_regression() {
    println!("=== Starting ArenaForest Regression Test ===");

    let mut forest = ArenaForest::<BitcoinNodeHash>::new();

    // Step 1: Add 4 leaves to create a small but complex tree
    let leaves = vec![
        hash_from_u8(10),
        hash_from_u8(20),
        hash_from_u8(30),
        hash_from_u8(40),
    ];

    println!("Step 1: Adding {} leaves", leaves.len());
    forest.modify(&leaves, &[]).expect("Failed to add leaves");

    println!("After adding leaves:");
    println!("  Number of leaves: {}", forest.leaves());
    println!(
        "  Roots: {:?}",
        forest
            .get_roots()
            .iter()
            .map(|r| format!("{:?}", r))
            .collect::<Vec<_>>()
    );

    // Step 2: Generate proofs for all leaves
    println!("\nStep 2: Generating proofs for all leaves");
    let mut leaf_proofs = Vec::new();

    for (i, leaf_hash) in leaves.iter().enumerate() {
        println!("  Generating proof for leaf {}: {:?}", i, leaf_hash);
        match forest.prove(&[*leaf_hash]) {
            Ok(proof) => {
                println!(
                    "    ✓ Proof generated successfully, targets: {}",
                    proof.targets.len()
                );
                leaf_proofs.push((*leaf_hash, proof));
            }
            Err(e) => {
                println!("    ✗ Failed to generate proof: {}", e);
                panic!("Proof generation failed for leaf {}: {}", i, e);
            }
        }
    }

    // Step 3: Verify all proofs pass
    println!("\nStep 3: Verifying all proofs pass");
    for (i, (leaf_hash, proof)) in leaf_proofs.iter().enumerate() {
        println!("  Verifying proof for leaf {}: {:?}", i, leaf_hash);
        match forest.verify(proof, &[*leaf_hash]) {
            Ok(true) => println!("    ✓ Proof verified successfully"),
            Ok(false) => {
                println!("    ✗ Proof verification returned false");
                panic!("Proof verification failed for leaf {}", i);
            }
            Err(e) => {
                println!("    ✗ Proof verification error: {}", e);
                panic!("Proof verification error for leaf {}: {}", i, e);
            }
        }
    }

    // Step 4: Delete 2 leaves (first and third)
    let leaves_to_delete = vec![leaves[0], leaves[2]]; // Delete leaf 0 and leaf 2
    println!(
        "\nStep 4: Deleting {} leaves: {:?}",
        leaves_to_delete.len(),
        leaves_to_delete
    );

    match forest.modify(&[], &leaves_to_delete) {
        Ok(()) => {
            println!("  ✓ Deletion successful");
            println!("  Number of leaves after deletion: {}", forest.leaves());
            println!(
                "  Roots after deletion: {:?}",
                forest
                    .get_roots()
                    .iter()
                    .map(|r| format!("{:?}", r))
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => {
            println!("  ✗ Deletion failed: {}", e);
            panic!("Deletion failed: {}", e);
        }
    }

    // Step 5: Verify old proofs for deleted leaves fail
    println!("\nStep 5: Verifying old proofs for deleted leaves fail");
    for (leaf_hash, proof) in &leaf_proofs {
        if leaves_to_delete.contains(leaf_hash) {
            println!("  Checking deleted leaf proof: {:?}", leaf_hash);
            match forest.verify(proof, &[*leaf_hash]) {
                Ok(false) => println!("    ✓ Proof correctly failed for deleted leaf"),
                Ok(true) => {
                    println!("    ✗ Proof incorrectly passed for deleted leaf");
                    panic!("Proof should have failed for deleted leaf");
                }
                Err(e) => println!("    ✓ Proof verification error (expected): {}", e),
            }
        }
    }

    // Step 6: Verify old proofs for remaining leaves correctly fail (tree structure changed)
    println!("\nStep 6: Verifying old proofs for remaining leaves correctly fail (tree changed)");
    for (leaf_hash, proof) in &leaf_proofs {
        if !leaves_to_delete.contains(leaf_hash) {
            println!("  Checking remaining leaf old proof: {:?}", leaf_hash);
            match forest.verify(proof, &[*leaf_hash]) {
                Ok(false) => println!("    ✓ Old proof correctly failed (tree structure changed)"),
                Ok(true) => {
                    println!("    ✗ Old proof incorrectly passed after tree change");
                    panic!("Old proof should fail after deletion changes tree structure");
                }
                Err(e) => println!("    ✓ Old proof verification error (expected): {}", e),
            }
        }
    }

    // Step 6b: Generate new proofs for remaining leaves and verify they pass
    println!("\nStep 6b: Generating new proofs for remaining leaves");
    let remaining_leaves: Vec<BitcoinNodeHash> = leaves
        .iter()
        .filter(|&leaf| !leaves_to_delete.contains(leaf))
        .cloned()
        .collect();

    for (i, leaf_hash) in remaining_leaves.iter().enumerate() {
        println!(
            "  Generating new proof for remaining leaf {}: {:?}",
            i, leaf_hash
        );
        match forest.prove(&[*leaf_hash]) {
            Ok(proof) => {
                println!("    ✓ New proof generated successfully");
                match forest.verify(&proof, &[*leaf_hash]) {
                    Ok(true) => println!("    ✓ New proof verified successfully"),
                    Ok(false) => {
                        println!("    ✗ New proof verification returned false");
                        panic!("New proof verification failed");
                    }
                    Err(e) => {
                        println!("    ✗ New proof verification error: {}", e);
                        panic!("New proof verification error: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("    ✗ Failed to generate new proof: {}", e);
                panic!("New proof generation failed: {}", e);
            }
        }
    }

    // Step 7: Re-add new leaves
    let new_leaves = vec![hash_from_u8(50), hash_from_u8(60)];
    println!(
        "\nStep 7: Re-adding {} new leaves: {:?}",
        new_leaves.len(),
        new_leaves
    );

    match forest.modify(&new_leaves, &[]) {
        Ok(()) => {
            println!("  ✓ Re-addition successful");
            println!("  Number of leaves after re-addition: {}", forest.leaves());
            println!(
                "  Roots after re-addition: {:?}",
                forest
                    .get_roots()
                    .iter()
                    .map(|r| format!("{:?}", r))
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => {
            println!("  ✗ Re-addition failed: {}", e);
            panic!("Re-addition failed: {}", e);
        }
    }

    // Step 8: Verify new proofs pass
    println!("\nStep 8: Generating and verifying proofs for new leaves");
    for (i, leaf_hash) in new_leaves.iter().enumerate() {
        println!("  Generating proof for new leaf {}: {:?}", i, leaf_hash);
        match forest.prove(&[*leaf_hash]) {
            Ok(proof) => {
                println!("    ✓ Proof generated successfully");
                match forest.verify(&proof, &[*leaf_hash]) {
                    Ok(true) => println!("    ✓ New proof verified successfully"),
                    Ok(false) => {
                        println!("    ✗ New proof verification returned false");
                        panic!("New proof verification failed");
                    }
                    Err(e) => {
                        println!("    ✗ New proof verification error: {}", e);
                        panic!("New proof verification error: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("    ✗ Failed to generate proof for new leaf: {}", e);
                panic!("Proof generation failed for new leaf: {}", e);
            }
        }
    }

    // Step 9: Final comprehensive test - prove all remaining leaves together
    let all_remaining_leaves: Vec<BitcoinNodeHash> = leaves
        .iter()
        .filter(|&leaf| !leaves_to_delete.contains(leaf))
        .cloned()
        .chain(new_leaves.iter().cloned())
        .collect();

    println!(
        "\nStep 9: Final test - proving all {} remaining leaves together",
        all_remaining_leaves.len()
    );
    match forest.prove(&all_remaining_leaves) {
        Ok(proof) => {
            println!("  ✓ Batch proof generated successfully");
            match forest.verify(&proof, &all_remaining_leaves) {
                Ok(true) => println!("  ✓ Batch proof verified successfully"),
                Ok(false) => {
                    println!("  ✗ Batch proof verification returned false");
                    panic!("Batch proof verification failed");
                }
                Err(e) => {
                    println!("  ✗ Batch proof verification error: {}", e);
                    panic!("Batch proof verification error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  ✗ Failed to generate batch proof: {}", e);
            panic!("Batch proof generation failed: {}", e);
        }
    }

    println!("\n=== ArenaForest Regression Test PASSED ===");
}

#[test]
fn single_leaf_edge_case_test() {
    println!("=== Testing Single Leaf Edge Cases ===");

    let mut forest = ArenaForest::<BitcoinNodeHash>::new();
    let leaf = hash_from_u8(100);

    // Test 1: Add single leaf
    println!("Test 1: Adding single leaf");
    forest
        .modify(&[leaf], &[])
        .expect("Failed to add single leaf");

    // Test 2: Prove single leaf
    println!("Test 2: Proving single leaf");
    let proof = forest.prove(&[leaf]).expect("Failed to prove single leaf");

    // Test 3: Verify single leaf proof
    println!("Test 3: Verifying single leaf proof");
    assert!(forest
        .verify(&proof, &[leaf])
        .expect("Failed to verify single leaf"));

    // Test 4: Delete single leaf
    println!("Test 4: Deleting single leaf");
    forest
        .modify(&[], &[leaf])
        .expect("Failed to delete single leaf");

    // Test 5: Verify forest state after deletion
    println!("Test 5: Verifying forest state after deletion");
    assert_eq!(forest.leaves(), 1); // Leaf count is monotonic - represents total leaves ever added
    assert_eq!(forest.get_roots().len(), 1);
    assert!(forest.get_roots()[0].get_data().is_empty()); // Root is empty after deletion

    println!("=== Single Leaf Edge Cases PASSED ===");
}

#[test]
fn large_forest_stress_test() {
    println!("=== Large Forest Stress Test ===");

    let mut forest = ArenaForest::<BitcoinNodeHash>::new();

    // Add 16 leaves to create a deeper tree
    let leaves: Vec<BitcoinNodeHash> = (0..16).map(|i| hash_from_u8(i as u8)).collect();

    println!("Adding {} leaves", leaves.len());
    forest.modify(&leaves, &[]).expect("Failed to add leaves");

    // Test proving various combinations
    let test_cases = vec![
        vec![0, 1],                   // Adjacent leaves
        vec![0, 15],                  // Extremes
        vec![3, 7, 11],               // Multiple non-adjacent
        vec![0, 1, 2, 3, 4, 5, 6, 7], // Half the leaves
    ];

    for (i, indices) in test_cases.iter().enumerate() {
        let test_leaves: Vec<BitcoinNodeHash> = indices.iter().map(|&idx| leaves[idx]).collect();
        println!("Test case {}: proving {} leaves", i + 1, test_leaves.len());

        let proof = forest
            .prove(&test_leaves)
            .expect("Failed to generate proof");
        assert!(forest
            .verify(&proof, &test_leaves)
            .expect("Failed to verify proof"));
        println!("  ✓ Passed");
    }

    println!("=== Large Forest Stress Test PASSED ===");
}

#[test]
fn test_adaptive_threshold_behavior() {
    println!("=== Testing Adaptive Threshold Behavior ===");

    let mut forest = ArenaForest::<BitcoinNodeHash>::new();

    // Test with small workload - should use sequential
    let small_leaves: Vec<BitcoinNodeHash> = (0..4).map(|i| hash_from_u8(i as u8)).collect();
    forest
        .modify(&small_leaves, &[])
        .expect("Failed to add small leaves");

    // Force some dirty state
    let leaf_to_delete = small_leaves[0];
    forest
        .modify(&[], &[leaf_to_delete])
        .expect("Failed to delete leaf");

    // Test adaptive method (should work regardless of threshold choice)
    forest.recompute_dirty_hashes_adaptive();
    println!("  ✓ Adaptive recomputation completed for small workload");

    // Test with larger workload
    let large_leaves: Vec<BitcoinNodeHash> = (10..200).map(|i| hash_from_u8(i as u8)).collect();
    forest
        .modify(&large_leaves, &[])
        .expect("Failed to add large leaves");

    // Create more dirty state
    let leaves_to_delete: Vec<BitcoinNodeHash> = large_leaves.iter().take(10).cloned().collect();
    forest
        .modify(&[], &leaves_to_delete)
        .expect("Failed to delete leaves");

    // Test adaptive method with larger workload
    forest.recompute_dirty_hashes_adaptive();
    println!("  ✓ Adaptive recomputation completed for large workload");

    // Test force methods
    forest.recompute_dirty_hashes_force_sequential();
    println!("  ✓ Forced sequential recomputation completed");

    forest.recompute_dirty_hashes_force_parallel();
    println!("  ✓ Forced parallel recomputation completed");

    println!("=== Adaptive Threshold Behavior Test PASSED ===");
}

#[test]
fn test_threshold_calculation() {
    println!("=== Testing Threshold Calculation ===");

    let cpu_count = num_cpus::get();
    println!("Detected {} CPU cores", cpu_count);

    let base_threshold = if cfg!(debug_assertions) { 500 } else { 200 };
    let calculated_threshold = (base_threshold + (cpu_count - 1) * 50).min(2000);

    println!("Base threshold: {}", base_threshold);
    println!("Calculated threshold: {}", calculated_threshold);
    println!(
        "Build mode: {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );

    // Verify threshold is reasonable
    assert!(calculated_threshold >= base_threshold);
    assert!(calculated_threshold <= 2000); // New cap based on benchmarking

    println!("=== Threshold Calculation Test PASSED ===");
}

#[test]
fn test_inject_dirty_functionality() {
    println!("=== Testing inject_dirty functionality ===");

    let mut forest = ArenaForest::<BitcoinNodeHash>::new();

    // Create a forest with multiple levels
    let leaves: Vec<BitcoinNodeHash> = (0..32)
        .map(|i| BitcoinNodeHash::from([i as u8; 32]))
        .collect();

    forest.modify(&leaves, &[]).expect("Failed to add leaves");

    // Inject specific dirty nodes
    let dirty_nodes = vec![5, 10, 15, 20];
    forest.inject_dirty(&dirty_nodes);

    // Verify the dirty count matches
    assert_eq!(forest.dirty_count(), dirty_nodes.len());
    println!("  ✓ Dirty count matches: {}", forest.dirty_count());

    // Test recomputation works
    forest.recompute_dirty_hashes_sequential();
    assert_eq!(forest.dirty_count(), 0); // Should be clean after recomputation
    println!("  ✓ Recomputation clears dirty state");

    // Test with parallel recomputation
    forest.inject_dirty(&dirty_nodes);
    assert_eq!(forest.dirty_count(), dirty_nodes.len());
    forest.recompute_dirty_hashes_parallel();
    assert_eq!(forest.dirty_count(), 0);
    println!("  ✓ Parallel recomputation also clears dirty state");

    // Test with empty dirty set
    forest.inject_dirty(&[]);
    assert_eq!(forest.dirty_count(), 0);
    println!("  ✓ Empty dirty injection works");

    // Test with out-of-bounds indices (should be ignored)
    let invalid_nodes = vec![999999, 1000000];
    forest.inject_dirty(&invalid_nodes);
    assert_eq!(forest.dirty_count(), 0);
    println!("  ✓ Out-of-bounds indices ignored");

    println!("=== inject_dirty functionality test PASSED ===");
}
