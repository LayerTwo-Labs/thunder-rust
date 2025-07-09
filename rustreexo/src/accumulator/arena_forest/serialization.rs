//! Serialization and deserialization for ArenaForest state persistence.

use std::collections::HashMap;
use std::io::Read;
use std::io::Write;

use smallvec::SmallVec;

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;
use super::types::ArenaNode;

impl<Hash: AccumulatorHash> ArenaForest<Hash> {
    /// Serialize the accumulator.
    pub fn serialize<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writer.write_all(&self.leaves.to_le_bytes())?;
        writer.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.max_height as u64).to_le_bytes())?;

        for node in &self.nodes {
            node.hash.write(&mut writer)?;
            writer.write_all(&node.lr.to_le_bytes())?;
            writer.write_all(&node.rr.to_le_bytes())?;
            writer.write_all(&node.parent.to_le_bytes())?;
            writer.write_all(&node.level.to_le_bytes())?;
        }

        writer.write_all(&(self.roots.len() as u64).to_le_bytes())?;
        for &root_idx in &self.roots {
            match root_idx {
                Some(idx) => {
                    writer.write_all(&[1])?;
                    writer.write_all(&idx.to_le_bytes())?;
                }
                None => {
                    writer.write_all(&[0])?;
                }
            }
        }

        writer.write_all(&(self.hash_to_node.len() as u64).to_le_bytes())?;
        for (&hash, &node_idx) in &self.hash_to_node {
            hash.write(&mut writer)?;
            writer.write_all(&node_idx.to_le_bytes())?;
        }

        Ok(())
    }

    /// Deserialize the accumulator.
    pub fn deserialize<R: Read>(mut reader: R) -> std::io::Result<ArenaForest<Hash>> {
        fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        }

        fn read_u32<R: Read>(reader: &mut R) -> std::io::Result<u32> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf))
        }

        let leaves = read_u64(&mut reader)?;
        let nodes_count = read_u64(&mut reader)?;
        let max_height = read_u64(&mut reader)? as usize;

        let mut nodes = Vec::with_capacity(nodes_count as usize);
        for _ in 0..nodes_count {
            let hash = Hash::read(&mut reader)?;
            let lr = read_u32(&mut reader)?;
            let rr = read_u32(&mut reader)?;
            let parent = read_u32(&mut reader)?;
            let level = read_u32(&mut reader)?;

            nodes.push(ArenaNode {
                hash,
                lr,
                rr,
                parent,
                level,
            });
        }

        let roots_count = read_u64(&mut reader)?;
        let mut roots = Vec::with_capacity(roots_count as usize);
        for _ in 0..roots_count {
            let mut has_root = [0u8; 1];
            reader.read_exact(&mut has_root)?;

            if has_root[0] == 1 {
                let idx = read_u32(&mut reader)?;
                roots.push(Some(idx));
            } else {
                roots.push(None);
            }
        }

        let mapping_count = read_u64(&mut reader)?;
        let mut hash_to_node = HashMap::with_capacity(mapping_count as usize);
        for _ in 0..mapping_count {
            let hash = Hash::read(&mut reader)?;
            let node_idx = read_u32(&mut reader)?;
            hash_to_node.insert(hash, node_idx);
        }

        // Initialize dirty levels
        let dirty_levels = vec![SmallVec::new()];

        Ok(ArenaForest {
            nodes,
            roots,
            leaves,
            hash_to_node,
            dirty_levels,
            max_height,
        })
    }
}
