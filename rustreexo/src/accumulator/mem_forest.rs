//! A full MemForest accumulator implementation. This is a simple version of the forest,
//! that keeps every node in memory. This is may require more memory, but is faster
//! to update, prove and verify.
//!
//! # Example
//! ```
//! use rustreexo::accumulator::mem_forest::MemForest;
//! use rustreexo::accumulator::node_hash::AccumulatorHash;
//! use rustreexo::accumulator::node_hash::BitcoinNodeHash;
//!
//! let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
//! let hashes: Vec<BitcoinNodeHash> = values
//!     .into_iter()
//!     .map(|i| BitcoinNodeHash::from([i; 32]))
//!     .collect();
//!
//! let mut p = MemForest::<BitcoinNodeHash>::new();
//!
//! p.modify(&hashes, &[]).expect("MemForest should not fail");
//! assert_eq!(p.get_roots().len(), 1);
//!
//! p.modify(&[], &hashes).expect("Still should not fail"); // Remove leaves from the accumulator
//!
//! assert_eq!(p.get_roots().len(), 1);
//! assert_eq!(p.get_roots()[0].get_data(), BitcoinNodeHash::default());
//! ```

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::io::Read;
use std::io::Write;
use std::rc::Rc;
use std::rc::Weak;

use super::node_hash::AccumulatorHash;
use super::node_hash::BitcoinNodeHash;
use super::proof::Proof;
use super::util::detect_offset;
use super::util::get_proof_positions;
use super::util::is_left_niece;
use super::util::is_root_populated;
use super::util::left_child;
use super::util::max_position_at_row;
use super::util::right_child;
use super::util::root_position;
use super::util::tree_rows;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
    Branch,
    Leaf,
}

// A few type aliases to improve readability.

/// A weak reference to a node in the forest. We use this when referencing to a node's
/// parent, as we don't want to create a reference cycle.
type Parent<T> = RefCell<Option<Weak<T>>>;
/// A reference to a node's children. We use this to store the left and right children
/// of a node. This will be the only long-lived strong reference to a node. If this gets
/// dropped, the node will be dropped as well.
type Children<T> = RefCell<Option<Rc<T>>>;

/// A forest node that can either be a leaf or a branch.
#[derive(Clone)]
pub struct Node<Hash: AccumulatorHash = BitcoinNodeHash> {
    /// The type of this node.
    ty: NodeType,
    /// The hash of the stored in this node.
    data: Cell<Hash>,
    /// The parent of this node, if any.
    parent: Parent<Node<Hash>>,
    /// The left and right children of this node, if any.
    left: Children<Node<Hash>>,
    /// The left and right children of this node, if any.
    right: Children<Node<Hash>>,
}

impl<Hash: AccumulatorHash> Node<Hash> {
    /// Writes one node to the writer, this method will recursively write all children.
    /// The primary use of this method is to serialize the accumulator. In this case,
    /// you should call this method on each root in the forest.
    pub fn write_one<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        match self.ty {
            NodeType::Branch => writer.write_all(&0_u64.to_le_bytes())?,
            NodeType::Leaf => writer.write_all(&1_u64.to_le_bytes())?,
        }
        self.data.get().write(writer)?;
        self.left
            .borrow()
            .as_ref()
            .map(|l| l.write_one(writer))
            .transpose()?;

        self.right
            .borrow()
            .as_ref()
            .map(|r| r.write_one(writer))
            .transpose()?;
        Ok(())
    }
    /// Recomputes the hash of all nodes, up to the root.
    fn recompute_hashes(&self) {
        let left = self.left.borrow();
        let right = self.right.borrow();

        if let (Some(left), Some(right)) = (left.as_deref(), right.as_deref()) {
            self.data
                .replace(Hash::parent_hash(&left.data.get(), &right.data.get()));
        }
        if let Some(ref parent) = *self.parent.borrow() {
            if let Some(p) = parent.upgrade() {
                p.recompute_hashes();
            }
        }
    }

    /// Reads one node from the reader, this method will recursively read all children.
    /// The primary use of this method is to deserialize the accumulator. In this case,
    /// you should call this method on each root in the forest, assuming you know how
    /// many roots there are.
    #[allow(clippy::type_complexity)]
    pub fn read_one<R: std::io::Read>(
        reader: &mut R,
    ) -> std::io::Result<(Rc<Node<Hash>>, HashMap<Hash, Weak<Node<Hash>>>)> {
        fn _read_one<Hash: AccumulatorHash, R: std::io::Read>(
            ancestor: Option<Rc<Node<Hash>>>,
            reader: &mut R,
            index: &mut HashMap<Hash, Weak<Node<Hash>>>,
        ) -> std::io::Result<Rc<Node<Hash>>> {
            let mut ty = [0u8; 8];
            reader.read_exact(&mut ty)?;
            let data = Hash::read(reader)?;

            let ty = match u64::from_le_bytes(ty) {
                0 => NodeType::Branch,
                1 => NodeType::Leaf,
                _ => panic!("Invalid node type"),
            };
            if ty == NodeType::Leaf {
                let leaf = Rc::new(Node {
                    ty,
                    data: Cell::new(data),
                    parent: RefCell::new(ancestor.map(|a| Rc::downgrade(&a))),
                    left: RefCell::new(None),
                    right: RefCell::new(None),
                });
                index.insert(leaf.data.get(), Rc::downgrade(&leaf));
                return Ok(leaf);
            }
            let node = Rc::new(Node {
                ty: NodeType::Branch,
                data: Cell::new(data),
                parent: RefCell::new(ancestor.map(|a| Rc::downgrade(&a))),
                left: RefCell::new(None),
                right: RefCell::new(None),
            });
            if !data.is_empty() {
                let left = _read_one(Some(node.clone()), reader, index)?;
                let right = _read_one(Some(node.clone()), reader, index)?;
                node.left.replace(Some(left));
                node.right.replace(Some(right));
            }
            node.left
                .borrow()
                .as_ref()
                .map(|l| l.parent.replace(Some(Rc::downgrade(&node))));
            node.right
                .borrow()
                .as_ref()
                .map(|r| r.parent.replace(Some(Rc::downgrade(&node))));

            Ok(node)
        }
        let mut index = HashMap::new();
        let root = _read_one(None, reader, &mut index)?;
        Ok((root, index))
    }

    /// Returns the data associated with this node.
    pub fn get_data(&self) -> Hash {
        self.data.get()
    }
}

/// The actual MemForest accumulator, it implements all methods required to update the forest
/// and to prove/verify membership.
#[derive(Default, Clone)]
pub struct MemForest<Hash: AccumulatorHash = BitcoinNodeHash> {
    /// The roots of the forest, all leaves are children of these roots, and therefore
    /// owned by them.
    roots: Vec<Rc<Node<Hash>>>,
    /// The number of leaves in the forest. Actually, this is the number of leaves we ever
    /// added to the forest.
    pub leaves: u64,
    /// A map of all nodes in the forest, indexed by their hash, this is used to lookup
    /// leaves when proving membership.
    map: HashMap<Hash, Weak<Node<Hash>>>,
}

impl MemForest {
    /// Creates a new empty [MemForest] with the default hash function.
    ///
    /// This will create an empty MemForest, using [BitcoinNodeHash] as the hash function. If you
    /// want to use a different hash function, you can use [MemForest::new_with_hash].
    /// # Example
    /// ```
    /// use rustreexo::accumulator::mem_forest::MemForest;
    /// let mut mem_forest = MemForest::new();
    /// ```
    pub fn new() -> MemForest {
        MemForest {
            map: HashMap::new(),
            roots: Vec::new(),
            leaves: 0,
        }
    }
}

impl<Hash: AccumulatorHash> MemForest<Hash> {
    /// Creates a new empty [MemForest] with a custom hash function.
    /// # Example
    /// ```
    /// use rustreexo::accumulator::mem_forest::MemForest;
    /// use rustreexo::accumulator::node_hash::BitcoinNodeHash;
    /// let mut MemForest = MemForest::<BitcoinNodeHash>::new();
    /// ```
    pub fn new_with_hash() -> MemForest<Hash> {
        MemForest {
            map: HashMap::new(),
            roots: Vec::new(),
            leaves: 0,
        }
    }

    /// Writes the MemForest to a writer. Used to send the accumulator over the wire
    /// or to disk.
    /// # Example
    /// ```
    /// use rustreexo::accumulator::mem_forest::MemForest;
    /// use rustreexo::accumulator::node_hash::BitcoinNodeHash;
    ///
    /// let mut mem_forest = MemForest::<BitcoinNodeHash>::new();
    /// let mut serialized = Vec::new();
    /// mem_forest.serialize(&mut serialized).unwrap();
    ///
    /// assert_eq!(
    ///     serialized,
    ///     vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    /// );
    /// ```
    pub fn serialize<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writer.write_all(&self.leaves.to_le_bytes())?;
        writer.write_all(&(self.roots.len() as u64).to_le_bytes())?;

        for root in &self.roots {
            root.write_one(&mut writer).unwrap();
        }

        Ok(())
    }

    /// Deserializes a MemForest from a reader.
    /// # Example
    /// ```
    /// use std::io::Cursor;
    ///
    /// use rustreexo::accumulator::mem_forest::MemForest;
    /// use rustreexo::accumulator::node_hash::BitcoinNodeHash;
    /// let mut serialized = Cursor::new(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    /// let MemForest = MemForest::<BitcoinNodeHash>::deserialize(&mut serialized).unwrap();
    /// assert_eq!(MemForest.leaves, 0);
    /// assert_eq!(MemForest.get_roots().len(), 0);
    /// ```
    pub fn deserialize<R: Read>(mut reader: R) -> std::io::Result<MemForest<Hash>> {
        fn read_u64<R: Read>(reader: &mut R) -> std::io::Result<u64> {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        }
        let leaves = read_u64(&mut reader)?;
        let roots_len = read_u64(&mut reader)?;
        let mut roots = Vec::new();
        let mut map = HashMap::new();
        for _ in 0..roots_len {
            let (root, _map) = Node::read_one(&mut reader)?;
            map.extend(_map);
            roots.push(root);
        }
        Ok(MemForest { roots, leaves, map })
    }

    /// Returns the hash of a given position in the tree.
    fn get_hash(&self, pos: u64) -> Result<Hash, String> {
        let (node, _, _) = self.grab_node(pos)?;
        Ok(node.data.get())
    }

    /// Proves that a given set of hashes is in the accumulator. It returns a proof
    /// and the hashes that we what to prove, but sorted by position in the tree.
    /// # Example
    /// ```
    /// use rustreexo::accumulator::mem_forest::MemForest;
    /// use rustreexo::accumulator::node_hash::BitcoinNodeHash;
    /// let mut mem_forest = MemForest::<BitcoinNodeHash>::new();
    /// let hashes = vec![0, 1, 2, 3, 4, 5, 6, 7]
    ///     .iter()
    ///     .map(|n| BitcoinNodeHash::from([*n; 32]))
    ///     .collect::<Vec<_>>();
    /// mem_forest.modify(&hashes, &[]).unwrap();
    /// // We want to prove that the first two hashes are in the accumulator.
    /// let proof = mem_forest.prove(&[hashes[1], hashes[0]]).unwrap();
    /// //TODO: Verify the proof
    /// ```
    pub fn prove(&self, targets: &[Hash]) -> Result<Proof<Hash>, String> {
        let mut positions = Vec::new();
        for target in targets {
            let node = self.map.get(target).ok_or("Could not find node")?;
            let position = self.get_pos(node)?;

            positions.push(position);
        }
        let needed = get_proof_positions(&positions, self.leaves, tree_rows(self.leaves));
        let proof = needed
            .iter()
            .map(|pos| self.get_hash(*pos).unwrap())
            .collect::<Vec<_>>();

        Ok(Proof::new_with_hash(positions, proof))
    }

    /// Returns a reference to the roots in this MemForest.
    pub fn get_roots(&self) -> &[Rc<Node<Hash>>] {
        &self.roots
    }

    /// Modify is the main API to a [MemForest]. Because order matters, you can only `modify`
    /// a [MemForest], and internally it'll add and delete, in the correct order.
    ///
    /// This method accepts two vectors as parameter, a vec of [Hash] and a vec of [u64]. The
    /// first one is a vec of leaf hashes for the newly created UTXOs. The second one is the position
    /// for the UTXOs being spent in this block as inputs.
    ///
    /// # Example
    /// ```
    /// use bitcoin_hashes::sha256::Hash as Data;
    /// use bitcoin_hashes::Hash;
    /// use bitcoin_hashes::HashEngine;
    /// use rustreexo::accumulator::mem_forest::MemForest;
    /// use rustreexo::accumulator::node_hash::BitcoinNodeHash;
    /// use rustreexo::accumulator::node_hash::NodeHash;
    /// let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
    /// let hashes = values
    ///     .into_iter()
    ///     .map(|val| {
    ///         let mut engine = Data::engine();
    ///         engine.input(&[val]);
    ///         BitcoinNodeHash::from(Data::from_engine(engine).as_byte_array())
    ///     })
    ///     .collect::<Vec<_>>();
    /// // Add 8 leaves to the MemForest
    /// let mut p = MemForest::<BitcoinNodeHash>::new();
    /// p.modify(&hashes, &[]).expect("MemForest should not fail");
    ///
    /// assert_eq!(
    ///     p.get_roots()[0].get_data().to_string(),
    ///     String::from("b151a956139bb821d4effa34ea95c17560e0135d1e4661fc23cedc3af49dac42")
    /// );
    /// ```
    pub fn modify(&mut self, add: &[Hash], del: &[Hash]) -> Result<(), String> {
        self.del(del)?;
        self.add(add);
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn grab_node(
        &self,
        pos: u64,
    ) -> Result<(Rc<Node<Hash>>, Rc<Node<Hash>>, Rc<Node<Hash>>), String> {
        let (tree, branch_len, bits) = detect_offset(pos, self.leaves);
        let mut n = Some(self.roots[tree as usize].clone());
        let mut sibling = Some(self.roots[tree as usize].clone());
        let mut parent = sibling.clone();

        for row in (0..(branch_len)).rev() {
            // Parent is the sibling of the current node as each of the
            // nodes point to their nieces.
            parent.clone_from(&sibling);

            // Figure out which node we need to follow.
            let niece_pos = ((bits >> row) & 1) as u8;

            #[allow(clippy::assigning_clones)]
            if let Some(node) = n {
                if is_left_niece(niece_pos as u64) {
                    n = node.right.borrow().clone();
                    sibling.clone_from(&*node.left.borrow());
                } else {
                    n = node.left.borrow().clone();
                    sibling.clone_from(&*node.right.borrow());
                }
            } else {
                sibling = None;
            }
        }
        if let (Some(node), Some(sibling), Some(parent)) = (n, sibling, parent) {
            return Ok((node, sibling, parent));
        }
        Err(format!("node {} not found", pos))
    }

    fn del(&mut self, targets: &[Hash]) -> Result<(), String> {
        let mut nodes = Vec::new();

        for target in targets {
            let node_ref = self.map.get(target).ok_or("Could not find node")?;
            let pos = self.get_pos(node_ref)?;

            let node = node_ref.upgrade().ok_or("Could not upgrade node")?;

            nodes.push((pos, node.get_data()));
        }

        nodes.sort_by(|a, b| a.0.cmp(&b.0));

        for (_, target) in nodes {
            match self.map.remove(&target) {
                Some(target) => {
                    self.del_single(&target.upgrade().unwrap());
                }
                None => {
                    return Err(format!("node {} not in the forest", target));
                }
            }
        }
        Ok(())
    }

    pub fn verify(&self, proof: &Proof<Hash>, del_hashes: &[Hash]) -> Result<bool, String> {
        let roots = self
            .roots
            .iter()
            .map(|root| root.get_data())
            .collect::<Vec<_>>();
        proof.verify(del_hashes, &roots, self.leaves)
    }

    fn get_pos(&self, node: &Weak<Node<Hash>>) -> Result<u64, String> {
        // This indicates whether the node is a left or right child at each level
        // When we go down the tree, we can use the indicator to know which
        // child to take.
        let mut left_child_indicator = 0_u64;
        let mut rows_to_top = 0;
        let mut node = node
            .upgrade()
            .ok_or("Could not upgrade node. Is this reference valid?")?;

        while let Some(parent) = node.parent.clone().into_inner() {
            let parent_left = parent
                .upgrade()
                .and_then(|parent| parent.left.clone().into_inner())
                .ok_or("Could not upgrade parent")?
                .clone();

            // If the current node is a left child, we left-shift the indicator
            // and leave the LSB as 0
            if parent_left.get_data() == node.get_data() {
                left_child_indicator <<= 1;
            } else {
                // If the current node is a right child, we left-shift the indicator
                // and set the LSB to 1
                left_child_indicator <<= 1;
                left_child_indicator |= 1;
            }
            rows_to_top += 1;
            node = parent.upgrade().ok_or("could not upgrade parent")?;
        }
        let mut root_idx = self.roots.len() - 1;
        let forest_rows = tree_rows(self.leaves);
        let mut root_row = None;
        // Find the root of the tree that the node belongs to
        for row in 0..=forest_rows {
            if is_root_populated(row, self.leaves) {
                let root = &self.roots[root_idx];
                if root.get_data() == node.get_data() {
                    root_row = Some(row);
                    break;
                }
                root_idx -= 1;
            }
        }

        let root_row = root_row.ok_or(format!(
            "Could not find the root position for row {root_idx}"
        ))?;
        let mut pos = root_position(self.leaves, root_row, forest_rows);
        for _ in 0..rows_to_top {
            // If LSB is 0, go left, otherwise go right
            match left_child_indicator & 1 {
                0 => {
                    pos = left_child(pos, forest_rows);
                }
                1 => {
                    pos = right_child(pos, forest_rows);
                }
                _ => unreachable!(),
            }
            left_child_indicator >>= 1;
        }

        Ok(pos)
    }

    fn del_single(&mut self, node: &Node<Hash>) -> Option<()> {
        let parent = node.parent.borrow();
        // Deleting a root
        let parent = match *parent {
            Some(ref node) => node.upgrade()?,
            None => {
                let pos = self.roots.iter().position(|x| x.data == node.data).unwrap();
                self.roots[pos] = Rc::new(Node {
                    ty: NodeType::Branch,
                    parent: RefCell::new(None),
                    data: Cell::new(Hash::empty()),
                    left: RefCell::new(None),
                    right: RefCell::new(None),
                });
                return None;
            }
        };

        let me = parent.left.borrow();
        // Can unwrap because we know the sibling exists
        let sibling = if me.as_deref()?.data == node.data {
            parent.right.borrow().clone()
        } else {
            parent.left.borrow().clone()
        };
        if let Some(ref sibling) = sibling {
            let grandparent = parent.parent.borrow().clone();
            sibling.parent.replace(grandparent.clone());

            if let Some(ref grandparent) = grandparent.and_then(|g| g.upgrade()) {
                if grandparent.left.borrow().clone().as_ref().unwrap().data == parent.data {
                    grandparent.left.replace(Some(sibling.clone()));
                } else {
                    grandparent.right.replace(Some(sibling.clone()));
                }
                sibling.recompute_hashes();
            } else {
                let pos = self
                    .roots
                    .iter()
                    .position(|x| x.data == parent.data)
                    .unwrap();
                self.roots[pos] = sibling.clone();
            }
        };

        Some(())
    }

    fn add_single(&mut self, value: Hash) {
        let mut node: Rc<Node<Hash>> = Rc::new(Node {
            ty: NodeType::Leaf,
            parent: RefCell::new(None),
            data: Cell::new(value),
            left: RefCell::new(None),
            right: RefCell::new(None),
        });
        self.map.insert(value, Rc::downgrade(&node));
        let mut leaves = self.leaves;
        while leaves & 1 != 0 {
            let root = self.roots.pop().unwrap();
            if root.get_data() == AccumulatorHash::empty() {
                leaves >>= 1;
                continue;
            }
            let new_node = Rc::new(Node {
                ty: NodeType::Branch,
                parent: RefCell::new(None),
                data: Cell::new(AccumulatorHash::parent_hash(
                    &root.data.get(),
                    &node.data.get(),
                )),
                left: RefCell::new(Some(root.clone())),
                right: RefCell::new(Some(node.clone())),
            });
            root.parent.replace(Some(Rc::downgrade(&new_node)));
            node.parent.replace(Some(Rc::downgrade(&new_node)));

            node = new_node;
            leaves >>= 1;
        }
        self.roots.push(node);
        self.leaves += 1;
    }

    fn add(&mut self, values: &[Hash]) {
        for value in values {
            self.add_single(*value);
        }
    }

    /// to_string returns the full MemForest in a string for all forests less than 6 rows.
    fn string(&self) -> String {
        if self.leaves == 0 {
            return "empty".to_owned();
        }
        let fh = tree_rows(self.leaves);
        // The accumulator should be less than 6 rows.
        if fh > 6 {
            let s = format!("Can't print {} leaves. roots: \n", self.leaves);
            return self.get_roots().iter().fold(s, |mut a, b| {
                a.extend(format!("{}\n", b.get_data()).chars());
                a
            });
        }
        let mut output = vec!["".to_string(); (fh as usize * 2) + 1];
        let mut pos: u8 = 0;
        for h in 0..=fh {
            let row_len = 1 << (fh - h);
            for _ in 0..row_len {
                let max = max_position_at_row(h, fh, self.leaves).unwrap();
                if max >= pos as u64 {
                    match self.get_hash(pos as u64) {
                        Ok(val) => {
                            if pos >= 100 {
                                output[h as usize * 2].push_str(
                                    format!("{:#02x}:{} ", pos, &val.to_string()[..2]).as_str(),
                                );
                            } else {
                                output[h as usize * 2].push_str(
                                    format!("{:0>2}:{} ", pos, &val.to_string()[..4]).as_str(),
                                );
                            }
                        }
                        Err(_) => {
                            output[h as usize * 2].push_str("        ");
                        }
                    }
                }

                if h > 0 {
                    output[(h as usize * 2) - 1].push_str("|-------");

                    for _ in 0..((1 << h) - 1) / 2 {
                        output[(h as usize * 2) - 1].push_str("--------");
                    }
                    output[(h as usize * 2) - 1].push_str("\\       ");

                    for _ in 0..((1 << h) - 1) / 2 {
                        output[(h as usize * 2) - 1].push_str("        ");
                    }

                    for _ in 0..(1 << h) - 1 {
                        output[h as usize * 2].push_str("        ");
                    }
                }
                pos += 1;
            }
        }

        output.iter().rev().fold(String::new(), |mut a, b| {
            a.push_str(b);
            a.push('\n');
            a
        })
    }
}

impl Debug for MemForest {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.string())
    }
}

impl Display for MemForest {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", self.string())
    }
}

#[cfg(test)]
mod test {
    use std::convert::TryFrom;
    use std::rc::Rc;
    use std::str::FromStr;
    use std::vec;

    use bitcoin_hashes::sha256::Hash as Data;
    use bitcoin_hashes::Hash;
    use bitcoin_hashes::HashEngine;
    use serde::Deserialize;

    use super::MemForest;
    use crate::accumulator::mem_forest::Node;
    use crate::accumulator::node_hash::AccumulatorHash;
    use crate::accumulator::node_hash::BitcoinNodeHash;
    use crate::accumulator::proof::Proof;

    fn hash_from_u8(value: u8) -> BitcoinNodeHash {
        let mut engine = Data::engine();

        engine.input(&[value]);

        BitcoinNodeHash::from(Data::from_engine(engine).as_byte_array())
    }

    #[test]
    fn test_grab_node() {
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let hashes = values.into_iter().map(hash_from_u8).collect::<Vec<_>>();

        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("MemForest should not fail");
        let (found_target, found_sibling, _) = p.grab_node(4).unwrap();
        let target = BitcoinNodeHash::try_from(
            "e52d9c508c502347344d8c07ad91cbd6068afc75ff6292f062a09ca381c89e71",
        )
        .unwrap();
        let sibling = BitcoinNodeHash::try_from(
            "e77b9a9ae9e30b0dbdb6f510a264ef9de781501d7b6b92ae89eb059c5ab743db",
        )
        .unwrap();

        assert_eq!(target, found_target.data.get());
        assert_eq!(sibling, found_sibling.data.get());
    }

    #[test]
    fn test_delete() {
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let hashes = values.into_iter().map(hash_from_u8).collect::<Vec<_>>();

        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("MemForest should not fail");
        p.modify(&[], &[hashes[0]]).expect("msg");

        let (node, _, _) = p.grab_node(8).unwrap();
        assert_eq!(
            String::from("4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"),
            node.data.get().to_string()
        );
    }

    #[test]
    fn test_proof_verify() {
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let hashes = values.into_iter().map(hash_from_u8).collect::<Vec<_>>();
        let mut p = MemForest::new();
        p.modify(&hashes, &[]).unwrap();

        let proof = p.prove(&[hashes[0], hashes[1]]).unwrap();
        assert!(p.verify(&proof, &[hashes[0], hashes[1]]).unwrap());
    }

    #[test]
    fn test_add() {
        let values = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let hashes = values.into_iter().map(hash_from_u8).collect::<Vec<_>>();

        let mut acc = MemForest::new();
        acc.add(&hashes);

        assert_eq!(
            "b151a956139bb821d4effa34ea95c17560e0135d1e4661fc23cedc3af49dac42",
            acc.roots[0].data.get().to_string().as_str(),
        );
        assert_eq!(
            "9c053db406c1a077112189469a3aca0573d3481bef09fa3d2eda3304d7d44be8",
            acc.roots[1].data.get().to_string().as_str(),
        );
        assert_eq!(
            "55d0a0ef8f5c25a9da266b36c0c5f4b31008ece82df2512c8966bddcc27a66a0",
            acc.roots[2].data.get().to_string().as_str(),
        );
        assert_eq!(
            "4d7b3ef7300acf70c892d8327db8272f54434adbc61a4e130a563cb59a0d0f47",
            acc.roots[3].data.get().to_string().as_str(),
        );
    }

    #[test]
    fn test_delete_roots_child() {
        // Assuming the following tree:
        //
        // 02
        // |---\
        // 00  01
        // If I delete `01`, then `00` will become a root, moving it's hash to `02`
        let values = vec![0, 1];
        let hashes: Vec<BitcoinNodeHash> = values.into_iter().map(hash_from_u8).collect();

        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("MemForest should not fail");
        p.del_single(&p.grab_node(1).unwrap().0);
        assert_eq!(p.get_roots().len(), 1);

        let root = p.get_roots()[0].clone();
        assert_eq!(root.data.get(), hashes[0]);
    }

    #[test]
    fn test_delete_root() {
        // Assuming the following tree:
        //
        // 02
        // |---\
        // 00  01
        // If I delete `02`, then `02` will become an empty root, it'll point to nothing
        // and its data will be Data::default()
        let values = vec![0, 1];
        let hashes: Vec<BitcoinNodeHash> = values.into_iter().map(hash_from_u8).collect();

        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("MemForest should not fail");
        p.del_single(&p.grab_node(2).unwrap().0);
        assert_eq!(p.get_roots().len(), 1);
        let root = p.get_roots()[0].clone();
        assert_eq!(root.data.get(), BitcoinNodeHash::default());
    }

    #[test]
    fn test_delete_non_root() {
        // Assuming this tree, if we delete `01`, 00 will move up to 08's position
        // 14
        // |-----------------\
        // 12                13
        // |-------\         |--------\
        // 08       09       10       11
        // |----\   |----\   |----\   |----\
        // 00   01  02   03  04   05  06   07

        // 14
        // |-----------------\
        // 12                13
        // |-------\         |--------\
        // 08       09       10       11
        // |----\   |----\   |----\   |----\
        // 00   01  02   03  04   05  06   07

        // Where 08's data is just 00's

        let values = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let hashes: Vec<BitcoinNodeHash> = values.into_iter().map(hash_from_u8).collect();

        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("MemForest should not fail");
        p.modify(&[], &[hashes[1]]).expect("Still should not fail");

        assert_eq!(p.roots.len(), 1);
        let (node, _, _) = p.grab_node(8).expect("This tree should have pos 8");
        assert_eq!(node.data.get(), hashes[0]);
    }

    #[derive(Debug, Deserialize)]
    struct TestCase {
        leaf_preimages: Vec<u8>,
        target_values: Option<Vec<u64>>,
        expected_roots: Vec<String>,
    }

    fn run_single_addition_case(case: TestCase) {
        let hashes = case
            .leaf_preimages
            .iter()
            .map(|preimage| hash_from_u8(*preimage))
            .collect::<Vec<_>>();
        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("Test mem_forests are valid");
        assert_eq!(p.get_roots().len(), case.expected_roots.len());
        let expected_roots = case
            .expected_roots
            .iter()
            .map(|root| BitcoinNodeHash::from_str(root).unwrap())
            .collect::<Vec<_>>();
        let roots = p
            .get_roots()
            .iter()
            .map(|root| root.data.get())
            .collect::<Vec<_>>();
        assert_eq!(expected_roots, roots, "Test case failed {:?}", case);
    }

    fn run_case_with_deletion(case: TestCase) {
        let hashes = case
            .leaf_preimages
            .iter()
            .map(|preimage| hash_from_u8(*preimage))
            .collect::<Vec<_>>();
        let dels = case
            .target_values
            .clone()
            .unwrap()
            .iter()
            .map(|pos| hashes[*pos as usize])
            .collect::<Vec<_>>();
        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("Test mem_forests are valid");
        p.modify(&[], &dels).expect("still should be valid");

        assert_eq!(p.get_roots().len(), case.expected_roots.len());
        let expected_roots = case
            .expected_roots
            .iter()
            .map(|root| BitcoinNodeHash::from_str(root).unwrap())
            .collect::<Vec<_>>();
        let roots = p
            .get_roots()
            .iter()
            .map(|root| root.data.get())
            .collect::<Vec<_>>();
        assert_eq!(expected_roots, roots, "Test case failed {:?}", case);
    }

    #[test]
    fn run_tests_from_cases() {
        #[derive(Deserialize)]
        struct TestsJSON {
            insertion_tests: Vec<TestCase>,
            deletion_tests: Vec<TestCase>,
        }

        let contents = std::fs::read_to_string("test_values/test_cases.json")
            .expect("Something went wrong reading the file");

        let tests = serde_json::from_str::<TestsJSON>(contents.as_str())
            .expect("JSON deserialization error");

        for i in tests.insertion_tests {
            run_single_addition_case(i);
        }
        for i in tests.deletion_tests {
            run_case_with_deletion(i);
        }
    }

    #[test]
    fn test_to_string() {
        let hashes = get_hash_vec_of(&(0..255).collect::<Vec<_>>());
        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("Test mem_forests are valid");
        assert_eq!(
            Some("Can't print 255 leaves. roots:"),
            p.to_string().get(0..30)
        );
    }

    #[test]
    fn test_get_pos() {
        macro_rules! test_get_pos {
            ($p:ident, $pos:literal) => {
                assert_eq!(
                    $p.get_pos(&Rc::downgrade(&$p.grab_node($pos).unwrap().0)),
                    Ok($pos)
                );
            };
        }
        let hashes = get_hash_vec_of(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("Test mem_forests are valid");
        test_get_pos!(p, 0);
        test_get_pos!(p, 1);
        test_get_pos!(p, 2);
        test_get_pos!(p, 3);
        test_get_pos!(p, 4);
        test_get_pos!(p, 5);
        test_get_pos!(p, 6);
        test_get_pos!(p, 7);
        test_get_pos!(p, 8);
        test_get_pos!(p, 9);
        test_get_pos!(p, 10);
        test_get_pos!(p, 11);
        test_get_pos!(p, 12);

        assert_eq!(p.get_pos(&Rc::downgrade(&p.get_roots()[0])), Ok(28));
        assert_eq!(
            p.get_pos(&Rc::downgrade(
                p.get_roots()[0].left.borrow().as_ref().unwrap()
            )),
            Ok(24)
        );
        assert_eq!(
            p.get_pos(&Rc::downgrade(
                p.get_roots()[0].right.borrow().as_ref().unwrap()
            )),
            Ok(25)
        );
    }

    #[test]
    fn test_serialize_one() {
        let hashes = get_hash_vec_of(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("Test mem_forests are valid");
        p.modify(&[], &[hashes[0]]).expect("can remove 0");
        let mut writer = std::io::Cursor::new(Vec::new());
        p.get_roots()[0].write_one(&mut writer).unwrap();
        let (deserialized, _) =
            Node::<BitcoinNodeHash>::read_one(&mut std::io::Cursor::new(writer.into_inner()))
                .unwrap();
        assert_eq!(deserialized.get_data(), p.get_roots()[0].get_data());
    }

    #[test]
    fn test_serialization() {
        let hashes = get_hash_vec_of(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("Test mem_forests are valid");
        p.modify(&[], &[hashes[0]]).expect("can remove 0");
        let mut writer = std::io::Cursor::new(Vec::new());
        p.serialize(&mut writer).unwrap();
        let deserialized = MemForest::<BitcoinNodeHash>::deserialize(&mut std::io::Cursor::new(
            writer.into_inner(),
        ))
        .unwrap();
        assert_eq!(
            deserialized.get_roots()[0].get_data(),
            p.get_roots()[0].get_data()
        );
        assert_eq!(deserialized.leaves, p.leaves);
        assert_eq!(deserialized.map.len(), p.map.len());
    }

    #[test]
    fn test_proof() {
        let hashes = get_hash_vec_of(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let del_hashes = [hashes[2], hashes[1], hashes[4], hashes[6]];

        let mut p = MemForest::new();
        p.modify(&hashes, &[]).expect("Test mem_forests are valid");

        let proof = p.prove(&del_hashes).expect("Should be able to prove");

        let expected_proof = Proof::new(
            [2, 1, 4, 6].to_vec(),
            vec![
                "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
                    .parse()
                    .unwrap(),
                "084fed08b978af4d7d196a7446a86b58009e636b611db16211b65a9aadff29c5"
                    .parse()
                    .unwrap(),
                "e77b9a9ae9e30b0dbdb6f510a264ef9de781501d7b6b92ae89eb059c5ab743db"
                    .parse()
                    .unwrap(),
                "ca358758f6d27e6cf45272937977a748fd88391db679ceda7dc7bf1f005ee879"
                    .parse()
                    .unwrap(),
            ],
        );
        assert_eq!(proof, expected_proof);
        assert!(p.verify(&proof, &del_hashes).unwrap());
    }

    fn get_hash_vec_of(elements: &[u8]) -> Vec<BitcoinNodeHash> {
        elements.iter().map(|el| hash_from_u8(*el)).collect()
    }

    #[test]
    fn test_display_empty() {
        let p = MemForest::new();
        let _ = p.to_string();
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut p = MemForest::<BitcoinNodeHash>::new();
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let hashes: Vec<BitcoinNodeHash> = values
            .into_iter()
            .map(|i| BitcoinNodeHash::from([i; 32]))
            .collect();
        p.modify(&hashes, &[]).expect("modify should work");
        assert_eq!(p.get_roots().len(), 1);
        assert!(!p.get_roots()[0].get_data().is_empty());
        assert_eq!(p.leaves, 16);
        p.modify(&[], &hashes).expect("modify should work");
        assert_eq!(p.get_roots().len(), 1);
        assert!(p.get_roots()[0].get_data().is_empty());
        assert_eq!(p.leaves, 16);
        let mut serialized = Vec::<u8>::new();
        p.serialize(&mut serialized).expect("serialize should work");
        let deserialized = MemForest::<BitcoinNodeHash>::deserialize(&*serialized)
            .expect("deserialize should work");
        assert_eq!(deserialized.get_roots().len(), 1);
        assert!(deserialized.get_roots()[0].get_data().is_empty());
        assert_eq!(deserialized.leaves, 16);
    }

    #[test]
    fn comprehensive_add_prove_delete_verify_regression_memforest() {
        println!("=== Starting MemForest Regression Test ===");

        let mut forest = MemForest::<BitcoinNodeHash>::new();

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
        println!("  Number of leaves: {}", forest.leaves);
        println!("  Number of roots: {}", forest.get_roots().len());
        for (i, root) in forest.get_roots().iter().enumerate() {
            println!("    Root {}: {:?}", i, root.get_data());
        }

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
                    println!("    ✓ Proof positions: {:?}", proof.targets);
                    println!("    ✓ Proof hashes count: {}", proof.hashes.len());
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
                println!("  Number of leaves after deletion: {}", forest.leaves);
                println!(
                    "  Number of roots after deletion: {}",
                    forest.get_roots().len()
                );
                for (i, root) in forest.get_roots().iter().enumerate() {
                    println!("    Root {}: {:?}", i, root.get_data());
                }
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
        println!(
            "\nStep 6: Verifying old proofs for remaining leaves correctly fail (tree changed)"
        );
        for (leaf_hash, proof) in &leaf_proofs {
            if !leaves_to_delete.contains(leaf_hash) {
                println!("  Checking remaining leaf old proof: {:?}", leaf_hash);
                match forest.verify(proof, &[*leaf_hash]) {
                    Ok(false) => {
                        println!("    ✓ Old proof correctly failed (tree structure changed)")
                    }
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
                    println!("    ✓ New proof positions: {:?}", proof.targets);
                    println!("    ✓ New proof hashes count: {}", proof.hashes.len());
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
                println!("  Number of leaves after re-addition: {}", forest.leaves);
                println!(
                    "  Number of roots after re-addition: {}",
                    forest.get_roots().len()
                );
                for (i, root) in forest.get_roots().iter().enumerate() {
                    println!("    Root {}: {:?}", i, root.get_data());
                }
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
                    println!("    ✓ Proof positions: {:?}", proof.targets);
                    println!("    ✓ Proof hashes count: {}", proof.hashes.len());
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
                println!("  ✓ Batch proof positions: {:?}", proof.targets);
                println!("  ✓ Batch proof hashes count: {}", proof.hashes.len());
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

        println!("\n=== MemForest Regression Test PASSED ===");
    }
}
