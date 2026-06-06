# oxide-compile-cache

Content-addressed compilation cache for GPU kernels. Hash source → lookup PTX → skip recompile. LRU eviction, TTL, hit rate tracking.

## Why This Matters

# oxide-compile-cache
Content-addressed compilation cache for GPU kernels.
Hash source → lookup PTX → skip recompile. LRU eviction, TTL.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub struct CacheEntry
pub struct CompileCache
pub fn new
pub fn hash_source
pub fn get
pub fn put
pub fn compile
pub fn invalidate
pub fn clear
pub fn hit_rate
pub fn time_saved_us
pub fn entry_count
```

## Usage

```toml
[dependencies]
oxide-compile-cache = "0.1.0"
```

```rust
use oxide_compile_cache::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-compile-cache.git
cd oxide-compile-cache
cargo test    # 8 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Rust | 177 |
| Public API | 14 items |

## License

Apache-2.0
