# oxide-compile-cache

Content-addressed compilation cache for GPU kernels with LRU eviction.

## Why This Exists

Kernel compilation is slow. A CUDA kernel that takes 50 μs to execute might take 500 ms to compile — a 10,000× overhead. When you're running the same kernels repeatedly (which is most GPU workloads), recompilation is pure waste. The fix is simple in concept: hash the source, check if you've already compiled it, return the cached PTX. The engineering challenge is doing this correctly — cache invalidation, eviction policy, hit rate tracking, and measuring actual time savings.

The content-addressed approach means cache keys are deterministic hashes of the source code. Same source → same hash → same cached PTX. No invalidation needed when the source changes — it produces a different hash and gets a fresh compilation. LRU eviction keeps the cache bounded. The `compile()` method abstracts the entire cache-or-compile decision into a single call.

## Architecture

```
┌────────────────────────────────────────────────┐
│              CompileCache                       │
│  max_entries: 128                               │
│  time_us: 50001                                 │
│  hits: 847  misses: 23                          │
│  time_saved_us: 8470000                         │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │ key: "a3f2b8c1..." → CacheEntry           │ │
│  │   ptx: [0x7f, 0x03, 0xb0, ...]           │ │
│  │   compile_time_us: 450000                 │ │
│  │   created_at: 1000                         │ │
│  │   last_accessed: 50000                     │ │
│  │   hits: 42                                 │ │
│  ├───────────────────────────────────────────┤ │
│  │ key: "7d9e4f2a..." → CacheEntry           │ │
│  │   ...                                      │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  compile(src, compile_fn) → (ptx, time)         │
│    ├─ cache hit → (cached_ptx, 0)               │
│    └─ cache miss → compile, store, (ptx, time)  │
│                                                 │
│  LRU Eviction:                                  │
│    On insert when full:                         │
│    Find entry with min(last_accessed)           │
│    Remove it, insert new entry                  │
└────────────────────────────────────────────────┘

Hash Function (DJB2 variant):
  h = 5381
  for byte in source: h = h * 33 + byte
  key = format!("{:016x}", h)
```

**Key types:**

- `CacheEntry` — key, PTX bytes, compile time, timestamps, hit count
- `CompileCache` — the cache engine with LRU eviction

## Usage

```rust
use oxide_oxide_compile_cache::CompileCache;

let mut cache = CompileCache::new(128); // max 128 entries

// Define a compilation function
fn compile_kernel(src: &str) -> (Vec<u8>, u64) {
    let ptx = vec![0x7f; src.len() * 10]; // simulated PTX
    let time = src.len() as u64 * 100;     // simulated compile time (μs)
    (ptx, time)
}

// First call: cache miss, compiles and stores
let (ptx, time) = cache.compile("kernel void add(...)", compile_kernel);
assert!(time > 0); // actually compiled
assert_eq!(cache.misses(), 1);

// Second call: cache hit, returns cached PTX
let (ptx, time) = cache.compile("kernel void add(...)", compile_kernel);
assert_eq!(time, 0); // instant — cache hit
assert_eq!(cache.hits(), 1);

// Check performance
println!("Hit rate: {:.1}%", cache.hit_rate() * 100.0);
println!("Time saved: {} ms", cache.time_saved_us() / 1000);

// Manual invalidation (e.g., compiler version changed)
cache.invalidate("kernel void add(...)");

// Direct get/put for custom compilation pipelines
if let Some(cached_ptx) = cache.get("my_kernel") {
    // use cached PTX
} else {
    let (ptx, time) = compile_kernel("my_kernel");
    cache.put("my_kernel", ptx, time);
}
```

## API Reference

### `CacheEntry`

```rust
pub struct CacheEntry {
    pub key: String,           // Content hash (16 hex chars)
    pub ptx: Vec<u8>,         // Compiled PTX bytecode
    pub compile_time_us: u64, // Original compilation time
    pub created_at: u64,      // Logical time of creation
    pub last_accessed: u64,   // Logical time of last access
    pub hits: u64,            // Number of cache hits
}
```

### `CompileCache`

- `new(max_entries: usize) -> Self`
- `compile(source, compile_fn) -> (Vec<u8>, u64)` — cache-or-compile, returns (PTX, compile_time)
- `get(source: &str) -> Option<Vec<u8>>` — cache lookup, updates access time
- `put(source, ptx, compile_time_us)` — insert entry with LRU eviction
- `invalidate(source: &str) -> bool` — remove specific entry
- `clear()` — remove all entries
- `hit_rate() -> f64` — hits / (hits + misses)
- `time_saved_us() -> u64` — total compile time saved by cache hits
- `entry_count() -> usize` / `hits() -> u64` / `misses() -> u64`
- `hash_source(source: &str) -> String` — (associated function) content hash

## The Deeper Idea

This is the **compilation layer** in the oxide stack's performance architecture. It sits between the optimization layer (oxide-gradient, which finds optimal parameters) and the execution layer (oxide-pipeline, which runs the compiled kernel). When oxide-gradient discovers that block_x=512, shared_mem=8192 produces the best occupancy, the compile cache ensures that this specific source+parameters combination is compiled once and reused indefinitely.

The content-addressed design means the cache is naturally correct — you can never get a stale PTX because the source *is* the key. The DJB2 hash is fast and sufficient for cache keys (it's not a security hash). The LRU eviction policy is optimal when the working set is larger than the cache, which is common in GPU workloads with many unique kernels but a hot subset that runs repeatedly.

## Related Crates

- **oxide-gradient** — produces optimized kernel parameters that the compile cache stores
- **oxide-pipeline** — executes compiled kernels, consults cache before recompilation
- **oxide-journal** — logs cache events (hits, misses, evictions) for audit trails
- **oxide-checkpoint** — can persist cache entries across restarts
