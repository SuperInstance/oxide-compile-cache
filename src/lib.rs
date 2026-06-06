//! # oxide-compile-cache
//!
//! Content-addressed compilation cache for GPU kernels.
//! Hash source → lookup PTX → skip recompile. LRU eviction, TTL.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub ptx: Vec<u8>,
    pub compile_time_us: u64,
    pub created_at: u64,
    pub last_accessed: u64,
    pub hits: u64,
}

pub struct CompileCache {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
    time_us: u64,
    hits: u64,
    misses: u64,
    time_saved_us: u64,
}

impl CompileCache {
    pub fn new(max_entries: usize) -> Self {
        Self { entries: HashMap::new(), max_entries, time_us: 0, hits: 0, misses: 0, time_saved_us: 0 }
    }

    fn advance(&mut self, delta: u64) { self.time_us += delta; }

    /// Hash source code to cache key (simplified).
    pub fn hash_source(source: &str) -> String {
        let mut hash: u64 = 5381;
        for b in source.bytes() { hash = hash.wrapping_mul(33).wrapping_add(b as u64); }
        format!("{:016x}", hash)
    }

    /// Look up cached PTX for source.
    pub fn get(&mut self, source: &str) -> Option<Vec<u8>> {
        let key = Self::hash_source(source);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_accessed = self.time_us;
            entry.hits += 1;
            self.hits += 1;
            self.time_saved_us += entry.compile_time_us;
            Some(entry.ptx.clone())
        } else { None }
    }

    /// Store compiled PTX in cache.
    pub fn put(&mut self, source: &str, ptx: Vec<u8>, compile_time_us: u64) {
        let key = Self::hash_source(source);
        self.advance(1);
        // Evict LRU if full
        if self.entries.len() >= self.max_entries {
            if let Some(lru_key) = self.entries.values()
                .min_by_key(|e| e.last_accessed)
                .map(|e| e.key.clone())
            { self.entries.remove(&lru_key); }
        }
        self.entries.insert(key.clone(), CacheEntry {
            key, ptx, compile_time_us, created_at: self.time_us,
            last_accessed: self.time_us, hits: 0,
        });
        self.misses += 1;
    }

    /// Compile with cache: hit returns cached, miss compiles and stores.
    pub fn compile(&mut self, source: &str, compile_fn: impl Fn(&str) -> (Vec<u8>, u64)) -> (Vec<u8>, u64) {
        if let Some(ptx) = self.get(source) {
            return (ptx, 0); // 0 compile time = cache hit
        }
        let (ptx, time) = compile_fn(source);
        self.put(source, ptx.clone(), time);
        (ptx, time)
    }

    pub fn invalidate(&mut self, source: &str) -> bool {
        let key = Self::hash_source(source);
        self.entries.remove(&key).is_some()
    }

    pub fn clear(&mut self) { self.entries.clear(); }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    pub fn time_saved_us(&self) -> u64 { self.time_saved_us }
    pub fn entry_count(&self) -> usize { self.entries.len() }
    pub fn hits(&self) -> u64 { self.hits }
    pub fn misses(&self) -> u64 { self.misses }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_compile(src: &str) -> (Vec<u8>, u64) {
        (vec![0x7f; src.len() * 10], src.len() as u64 * 100)
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = CompileCache::new(10);
        let (ptx, time) = cache.compile("kernel_a", mock_compile);
        assert!(!ptx.is_empty());
        assert!(time > 0);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = CompileCache::new(10);
        cache.compile("kernel_b", mock_compile);
        let (ptx, time) = cache.compile("kernel_b", mock_compile);
        assert!(!ptx.is_empty());
        assert_eq!(time, 0); // cache hit = 0 compile time
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = CompileCache::new(10);
        cache.compile("k1", mock_compile); // miss
        cache.compile("k1", mock_compile); // hit
        cache.compile("k1", mock_compile); // hit
        assert!((cache.hit_rate() - 0.667).abs() < 0.05);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = CompileCache::new(2);
        cache.compile("a", mock_compile);
        cache.compile("b", mock_compile);
        cache.compile("c", mock_compile); // should evict oldest
        assert_eq!(cache.entry_count(), 2);
    }

    #[test]
    fn test_invalidation() {
        let mut cache = CompileCache::new(10);
        cache.compile("target", mock_compile);
        assert!(cache.invalidate("target"));
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn test_time_saved() {
        let mut cache = CompileCache::new(10);
        cache.compile("k", mock_compile); // miss, time=100
        cache.compile("k", mock_compile); // hit, saved=100
        assert_eq!(cache.time_saved_us(), 100);
    }

    #[test]
    fn test_hash_deterministic() {
        let h1 = CompileCache::hash_source("hello");
        let h2 = CompileCache::hash_source("hello");
        assert_eq!(h1, h2);
        let h3 = CompileCache::hash_source("world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_clear() {
        let mut cache = CompileCache::new(10);
        cache.compile("x", mock_compile);
        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }
}
