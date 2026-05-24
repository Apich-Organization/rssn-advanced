//! Compact union-find (disjoint set union) over `u32` indices.
//!
//! Uses **path compression** (halving variant) and **union by rank** to
//! achieve amortised near-constant time for each find/union. Designed to
//! alias over `DagNodeId` values — callers cast to/from `u32`.

/// Path-compressed, rank-balanced union-find.
///
/// Each element `i` starts as its own root. `find` returns the canonical
/// representative of `i`'s equivalence class; `union` merges two classes.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    /// Creates a new `UnionFind` with `n` singleton classes `{0}, {1}, …`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    /// Pre-allocates capacity for `additional` more elements beyond the
    /// current length, avoiding reallocation during saturation when many
    /// new nodes are created by rewrite rules.
    pub fn reserve(&mut self, additional: usize) {
        self.parent.reserve(additional);
        self.rank.reserve(additional);
    }

    /// Extends the union-find to hold `n` total elements, adding new
    /// singletons for any indices that do not yet exist.
    pub fn grow_to(&mut self, n: usize) {
        let old = self.parent.len();
        if n <= old {
            return;
        }
        self.parent.extend(old as u32..n as u32);
        self.rank.resize(n, 0);
    }

    /// Returns the canonical representative of the class containing `x`.
    ///
    /// Uses path-halving: every other node on the path to root has its
    /// parent pointer set to its grandparent, halving the path length in
    /// amortised O(α(n)) per operation.
    pub fn find(&mut self, mut x: u32) -> u32 {
        // Bounds check: treat out-of-range IDs as their own root.
        if x as usize >= self.parent.len() {
            return x;
        }
        while self.parent[x as usize] != x {
            // Path halving: point to grandparent.
            let gp = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    /// Merges the equivalence classes of `x` and `y`.
    ///
    /// The higher-rank root becomes the new representative, preserving
    /// O(log n) tree height. Returns the canonical representative of the
    /// merged class.
    pub fn union(&mut self, x: u32, y: u32) -> u32 {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return rx;
        }

        // Grow if necessary.
        let max_idx = (rx.max(ry) as usize).saturating_add(1);
        self.grow_to(max_idx);

        let (root, child) = if self.rank[rx as usize] >= self.rank[ry as usize] {
            (rx, ry)
        } else {
            (ry, rx)
        };
        self.parent[child as usize] = root;
        if self.rank[rx as usize] == self.rank[ry as usize] {
            self.rank[root as usize] = self.rank[root as usize].saturating_add(1);
        }
        root
    }

    /// Returns `true` if `x` and `y` are in the same equivalence class.
    pub fn same(&mut self, x: u32, y: u32) -> bool {
        self.find(x) == self.find(y)
    }

    /// Number of elements tracked (not number of distinct classes).
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Returns `true` if no elements have been added yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_classes() {
        let mut uf = UnionFind::new(5);
        for i in 0..5u32 {
            assert_eq!(uf.find(i), i);
        }
    }

    #[test]
    fn union_merges_classes() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(2, 3);
        assert!(uf.same(0, 1));
        assert!(uf.same(2, 3));
        assert!(!uf.same(0, 2));
        uf.union(1, 2);
        assert!(uf.same(0, 3));
    }

    #[test]
    fn path_compression_gives_flat_structure() {
        let mut uf = UnionFind::new(8);
        // Chain: 0→1→2→3→4→5→6→7
        for i in 0..7u32 {
            uf.union(i, i + 1);
        }
        // After find, all should share the same root.
        let root = uf.find(0);
        for i in 0..8u32 {
            assert_eq!(uf.find(i), root);
        }
    }

    #[test]
    fn grow_to_extends_cleanly() {
        let mut uf = UnionFind::new(3);
        uf.grow_to(8);
        assert_eq!(uf.len(), 8);
        for i in 3..8u32 {
            assert_eq!(uf.find(i), i, "new singletons");
        }
    }
}
