//! Serialization and deserialization for ArenaForest state persistence.

use std::io::Read;
use std::io::Write;

use bitvec::vec::BitVec;
use hashbrown::HashMap;
use smallvec::SmallVec;

use super::super::node_hash::AccumulatorHash;
use super::forest::ArenaForest;

impl<Hash: AccumulatorHash + std::ops::Deref<Target = [u8; 32]>> ArenaForest<Hash> {
    /// Serialize the accumulator.
    pub fn serialize<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writer.write_all(&self.leaves.to_le_bytes())?;
        writer.write_all(&(self.hashes.len() as u64).to_le_bytes())?;
        writer.write_all(&(self.max_height as u64).to_le_bytes())?;

        for i in 0..self.hashes.len() {
            self.hashes[i].write(&mut writer)?;
            writer.write_all(&self.lr[i].to_le_bytes())?;
            writer.write_all(&self.rr[i].to_le_bytes())?;
            writer.write_all(&self.parent[i].to_le_bytes())?;
            writer.write_all(&self.level[i].to_le_bytes())?;
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

        let total_entries: usize = self
            .hash_to_node
            .values()
            .map(|entries| entries.len())
            .sum();
        writer.write_all(&(total_entries as u64).to_le_bytes())?;

        for entries in self.hash_to_node.values() {
            for (hash, node_idx) in entries {
                hash.write(&mut writer)?;
                writer.write_all(&node_idx.to_le_bytes())?;
            }
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

        let mut hashes = Vec::with_capacity(nodes_count as usize);
        let mut lr = Vec::with_capacity(nodes_count as usize);
        let mut rr = Vec::with_capacity(nodes_count as usize);
        let mut parent = Vec::with_capacity(nodes_count as usize);
        let mut level = Vec::with_capacity(nodes_count as usize);

        for _ in 0..nodes_count {
            let hash = Hash::read(&mut reader)?;
            let lr_val = read_u32(&mut reader)?;
            let rr_val = read_u32(&mut reader)?;
            let parent_val = read_u32(&mut reader)?;
            let level_val = read_u32(&mut reader)?;

            hashes.push(hash);
            lr.push(lr_val);
            rr.push(rr_val);
            parent.push(parent_val);
            level.push(level_val);
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
        let mut hash_to_node = HashMap::new();
        for _ in 0..mapping_count {
            let hash = Hash::read(&mut reader)?;
            let node_idx = read_u32(&mut reader)?;
            let key = super::forest::hash_key(&hash);
            hash_to_node
                .entry(key)
                .or_insert_with(SmallVec::new)
                .push((hash, node_idx));
        }

        let dirty = vec![BitVec::new()];

        Ok(ArenaForest {
            hashes,
            lr,
            rr,
            parent,
            level,
            roots,
            leaves,
            hash_to_node,
            dirty,
            max_height,
            zombies: super::types::ZombieQueue::new(100),
            dirty_queue: super::types::DirtyQueue::new(),
            root_levels: Vec::new(),
            root_cache: Vec::new(),
        })
    }
}
