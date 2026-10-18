// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 The Viv authors
//! The IR substrate: structural identity (1.3), mandatory metadata (1.2), and
//! content-addressed storage.
//!
//! The kit stays generic over a language's node *vocabulary*: a language defines
//! its own [`NodeKind`] and the kit computes identity, stores nodes, and walks
//! them without ever knowing what the variants mean. The lowering machinery
//! (1.4 stratified traversal, 1.5 lowering registry) is intentionally deferred
//! until a real lowering needs it; only the `LoweringChain` *field* is committed
//! now, so it never has to be retrofitted onto every node (1.2).

use std::collections::HashMap;
use std::fmt;

use crate::span::Span;

/// The hasher languages feed in [`NodeKind::hash_payload`]. Re-exported so a
/// language never takes a direct dependency on the hashing crate — the kit owns
/// the choice of hash (1.3) and can change it without touching any language.
pub use blake3::Hasher;

/// A node's structural identity (1.3): a BLAKE3 hash over its variant tag, its
/// literal payload, and its children's `NodeId`s.
///
/// Span and lineage are *not* hashed, so structurally identical nodes share an
/// id no matter where they appear — the corollary that a common subexpression
/// has exactly one identity (enabling GVN, caching, content-addressed storage).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId([u8; 32]);

impl NodeId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Short hex prefix, for display and debugging.
    pub fn short(&self) -> String {
        use fmt::Write;
        let mut s = String::with_capacity(8);
        for b in &self.0[..4] {
            let _ = write!(s, "{:02x}", b);
        }
        s
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.short())
    }
}

/// A language's IR vocabulary. The kit is a library, not a framework (2.1): a
/// language supplies the variant set and tells the kit how to hash each
/// variant's *intrinsic* content. The kit supplies everything else.
pub trait NodeKind: Clone {
    /// A discriminant, stable across runs, distinguishing this variant from its
    /// siblings for hashing purposes.
    fn tag(&self) -> u32;

    /// Feed this node's intrinsic content — literal values, names, type tags;
    /// everything that distinguishes it from a sibling of the same variant, but
    /// *not* its children — into the hasher. Children are hashed by the store.
    fn hash_payload(&self, hasher: &mut Hasher);
}

/// The sequence of transformations that produced a node (1.2). Empty for nodes
/// minted directly from source.
#[derive(Clone, Debug, Default)]
pub struct LoweringChain(pub Vec<LoweringStep>);

#[derive(Clone, Debug)]
pub struct LoweringStep {
    /// The content-addressed rule that fired (1.5).
    pub rule: NodeId,
    /// The higher-level node this one was lowered from.
    pub from: NodeId,
}

impl LoweringChain {
    pub fn empty() -> Self {
        Self(Vec::new())
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A stored IR node. Every node carries the three mandatory fields (1.2):
/// structural `id`, source `span`, and `lineage`. The only way to mint one is
/// [`IrProgram::add`], which computes the id and demands a span — so a language
/// physically cannot define a node that omits them.
#[derive(Clone, Debug)]
pub struct Node<K: NodeKind> {
    pub id: NodeId,
    pub span: Span,
    pub lineage: LoweringChain,
    pub kind: K,
    /// Child references, by `NodeId` — never by pointer or index (1.3).
    pub children: Vec<NodeId>,
}

/// A content-addressed store of IR nodes plus the program's roots. Nodes are
/// keyed by `NodeId`; inserting a structurally identical node returns the
/// existing id.
#[derive(Debug)]
pub struct IrProgram<K: NodeKind> {
    nodes: HashMap<NodeId, Node<K>>,
    roots: Vec<NodeId>,
}

impl<K: NodeKind> Default for IrProgram<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: NodeKind> IrProgram<K> {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), roots: Vec::new() }
    }

    /// Mint (or dedup) a node, computing its structural id from the kind's tag,
    /// the kind's payload, and the children's ids (1.3). The span is mandatory.
    ///
    /// Dedup note (a known tension): if a structurally identical node already
    /// exists, the *first* node's span wins and this `span` is dropped. For v1
    /// this is acceptable — codegen is unaffected and common subexpressions are
    /// rare in small programs. Recovering per-use spans is deferred; see the
    /// NodeId-vs-metadata discussion in the design notes.
    pub fn add(&mut self, kind: K, span: Span, children: Vec<NodeId>) -> NodeId {
        let id = self.compute_id(&kind, &children);
        self.nodes.entry(id).or_insert_with(|| Node {
            id,
            span,
            lineage: LoweringChain::empty(),
            kind,
            children,
        });
        id
    }

    fn compute_id(&self, kind: &K, children: &[NodeId]) -> NodeId {
        let mut h = blake3::Hasher::new();
        h.update(&kind.tag().to_le_bytes());
        kind.hash_payload(&mut h);
        // Child order is significant: length-prefix, then each id in order.
        h.update(&(children.len() as u32).to_le_bytes());
        for c in children {
            h.update(c.as_bytes());
        }
        NodeId(*h.finalize().as_bytes())
    }

    /// Look up a node. Panics if the id is foreign to this program — within one
    /// `IrProgram`, every id handed out by `add` is always present.
    pub fn get(&self, id: NodeId) -> &Node<K> {
        self.nodes
            .get(&id)
            .expect("NodeId not present in this IrProgram")
    }

    pub fn try_get(&self, id: NodeId) -> Option<&Node<K>> {
        self.nodes.get(&id)
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    pub fn add_root(&mut self, id: NodeId) {
        self.roots.push(id);
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
