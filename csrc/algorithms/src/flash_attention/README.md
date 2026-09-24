# Flash Attention kernel notes

This document describes the educational forward kernel in
[`flash_attention.cu`](./flash_attention.cu). The implementation uses tiled
Q/K/V staging and online softmax, but favors readability over peak performance.

## Input layout

The operator expects contiguous query, key, and value tensors with identical
shapes:

```text
[batch, heads, sequence_length, head_dim]
```

The output has the same shape. Supported head dimensions are 16, 32, 64, and
128.

The model hidden dimension does not appear directly in the kernel. For regular
multi-head attention:

```text
hidden_dim = heads * head_dim
```

The Q/K/V projections reshape `[batch, sequence_length, hidden_dim]` into the
layout above before calling this operator.

## Compile-time configuration (lines 20-23)

```cpp
constexpr int WARP_SIZE = 32;
constexpr int QUERY_ROWS_PER_BLOCK = 8;
constexpr int KEYS_PER_TILE = 32;
constexpr int FLASH_ATTENTION_THREADS = QUERY_ROWS_PER_BLOCK * WARP_SIZE;
```

- `WARP_SIZE` is the hardware warp width.
- `QUERY_ROWS_PER_BLOCK` assigns eight query rows to each block.
- `KEYS_PER_TILE` streams K and V in groups of 32 sequence positions.
- `FLASH_ATTENTION_THREADS` gives eight warps, or 256 threads, per block.

## Shared-memory layout (lines 25-30)

```cpp
template <typename T, int HEAD_DIM> struct FlashAttentionSharedStorage {
  T query[QUERY_ROWS_PER_BLOCK][HEAD_DIM];
  T key[KEYS_PER_TILE][HEAD_DIM];
  T value[KEYS_PER_TILE][HEAD_DIM];
  float probabilities[QUERY_ROWS_PER_BLOCK][KEYS_PER_TILE];
};
```

- `query` holds eight Q rows and stays resident for the block's lifetime.
- `key` holds the current 32-row K tile.
- `value` holds the corresponding 32-row V tile.
- `probabilities` holds one unnormalized probability per query/key pair in the
  current tile.
- The Q/K/V storage uses the input dtype. Probabilities and accumulators use
  fp32.
- The 16-byte alignment leaves room for adding vectorized loads later.

## Grid, block, warp, and lane mapping (lines 47-61 and 176-183)

The launcher creates:

```text
grid.x = ceil(sequence_length / 8)
grid.y = num_heads
grid.z = batch_size

block.x = 256 threads = 8 warps
```

Inside the kernel:

```text
blockIdx.z -> batch index
blockIdx.y -> attention-head index
blockIdx.x -> tile of eight query rows
warp_id    -> one query row in that tile
lane_id    -> one key in the current 32-key tile
```

The relevant calculations are:

```cpp
const auto block = cg::this_thread_block();
const auto warp = cg::tiled_partition<32>(block);

const int thread_id = block.thread_rank();
const int warp_id = warp.meta_group_rank();
const int lane_id = warp.thread_rank();

const int64_t query_tile_start = blockIdx.x * QUERY_ROWS_PER_BLOCK;
const int64_t query_index = query_tile_start + warp_id;
```

The flattened batch/head offset is:

```cpp
const int64_t head_offset =
    (blockIdx.z * num_heads + blockIdx.y) * sequence_length * HEAD_DIM;
```

For `[B=2, H=4, S=128, D=64]`, the launch has:

```text
grid             = (16, 4, 2) = 128 blocks
query rows/block = 8
warps/block      = 8
threads/block    = 256
K/V tile passes = 128 / 32 = 4
outputs/lane     = 64 / 32 = 2
```

## Kernel walkthrough (lines 41-168)

### 1. Create Cooperative Groups (lines 47-48)

```cpp
const auto block = cg::this_thread_block();
const auto warp = cg::tiled_partition<WARP_SIZE>(block);
```

`block` represents all 256 threads. `warp` represents the current thread's
32-thread tile. The kernel uses `block.sync()`, `warp.sync()`, and
`cg::reduce()` rather than raw synchronization and shuffle intrinsics.

### 2. Assign output dimensions to lanes (lines 52-55)

```cpp
constexpr int OUTPUTS_PER_LANE = utils::ceil_div(HEAD_DIM, WARP_SIZE);
```

Each lane owns output dimensions separated by the warp size:

```text
lane 0 -> dimensions 0, 32, 64, 96
lane 1 -> dimensions 1, 33, 65, 97
...
```

For `HEAD_DIM=128`, every lane owns four output values.

### 3. Load the Q tile (lines 63-74)

```cpp
constexpr int QUERY_TILE_ELEMENTS = QUERY_ROWS_PER_BLOCK * HEAD_DIM;
for (int index = thread_id; index < QUERY_TILE_ELEMENTS;
     index += FLASH_ATTENTION_THREADS) {
  const int row = index / HEAD_DIM;
  const int dim = index % HEAD_DIM;
  const int64_t global_row = query_tile_start + row;
  shared.query[row][dim] =
      global_row < sequence_length
          ? query[head_offset + global_row * HEAD_DIM + dim]
          : T{0};
}
```

All 256 threads cooperatively copy the eight Q rows into shared memory. Rows
past the end of a partial final tile are zero-filled. Q is loaded only once
because it is reused against every K/V tile.

The following `block.sync()` makes the complete Q tile visible to every warp.

### 4. Initialize online-softmax state (lines 76-83)

Each warp maintains the state for its query row:

```cpp
float row_max = -INFINITY;
float row_sum = 0.0f;
float output_accumulator[OUTPUTS_PER_LANE] = {0};
```

- `row_max` is the largest score seen in all completed K tiles.
- `row_sum` is the softmax denominator relative to `row_max`.
- `output_accumulator` contains the unnormalized partial `P @ V` result.
- The accumulator array is private per thread and normally resides in
  registers.

### 5. Stream over K/V tiles (lines 85-86)

```cpp
for (int64_t key_tile_start = 0;
     key_tile_start < sequence_length;
     key_tile_start += KEYS_PER_TILE) {
```

For `sequence_length=100`, the loop processes:

```text
tile 0: keys  0-31
tile 1: keys 32-63
tile 2: keys 64-95
tile 3: keys 96-99 plus zero padding
```

Only the current K/V tile is held in shared memory, so shared-memory usage does
not grow with sequence length.

### 6. Load a K/V tile (lines 87-104)

```cpp
constexpr int KEY_VALUE_TILE_ELEMENTS = KEYS_PER_TILE * HEAD_DIM;
for (int index = thread_id; index < KEY_VALUE_TILE_ELEMENTS;
     index += FLASH_ATTENTION_THREADS) {
```

The complete block cooperatively loads `32 * HEAD_DIM` K elements and the same
number of V elements. Out-of-range positions in the final tile are set to zero.

`block.sync()` is required because every warp subsequently reads the same K/V
tile.

### 7. Compute the QK transpose tile (lines 106-118)

```cpp
const int64_t key_index = key_tile_start + lane_id;
```

Each lane owns one key row. Each warp owns one query row, so each warp produces
32 scores:

```text
score[lane] = dot(Q[query_index], K[key_tile_start + lane]) * scale
```

The current dot product loops serially over `HEAD_DIM`. This is clear but not
fast; a production implementation would distribute the matrix multiply across
threads and use tensor cores.

For causal attention, a score is valid only when:

```text
key_index <= query_index
```

Invalid and padded positions receive negative infinity before softmax.

### 8. Compute the tile maximum (lines 120-124)

```cpp
const float tile_max = cg::reduce(warp, score, cg::greater<float>());
const float new_row_max = fmaxf(row_max, tile_max);
```

`cg::reduce` finds the largest of the warp's 32 scores. `new_row_max` combines
that result with the maximum from all previous tiles.

### 9. Correct the previous online-softmax state (lines 125-126)

```cpp
const float old_row_scale =
    row_sum == 0.0f ? 0.0f : expf(row_max - new_row_max);
```

If a later tile contains a larger score, previously accumulated values were
normalized relative to the old maximum. Multiplying by `old_row_scale`
converts them to the new maximum's coordinate system.

The zero-denominator condition handles the first tile without evaluating
`exp(-infinity - -infinity)`.

### 10. Compute tile probabilities and update state (lines 127-136)

```cpp
const float probability =
    score_is_valid ? expf(score - new_row_max) : 0.0f;
const float tile_sum = cg::reduce(warp, probability, cg::plus<float>());
```

These probabilities are deliberately not normalized yet. The thread-local
online-softmax state is advanced immediately after the probability reduction:

```cpp
if (query_is_valid) {
  row_max = new_row_max;
  row_sum = row_sum * old_row_scale + tile_sum;
}
```

Each lane then writes its probability to:

```cpp
shared.probabilities[warp_id][lane_id]
```

`warp.sync()` ensures all 32 values are visible before the warp computes its
`P @ V` contribution.

### 11. Accumulate the PV tile (lines 138-156)

For every output dimension owned by the lane:

```text
tile_output[d] = sum(probability[key] * V[key, d])
```

The previous result is then corrected and extended:

```cpp
output_accumulator =
    output_accumulator * old_row_scale + tile_output;
```

The final `block.sync()` ensures every warp has finished reading shared K/V
before the next loop iteration overwrites those arrays.

Invalid query warps must still participate in every block synchronization.
Returning early from those warps would deadlock a partially filled final
query tile.

### 12. Normalize and store the result (lines 159-167)

After every K/V tile has been processed:

```cpp
output[...] = static_cast<T>(output_accumulator[...] / row_sum);
```

Conceptually, this produces:

```text
    sum(exp(score - max) * V)
O = -------------------------
        sum(exp(score - max))
```

The fp32 result is converted back to the input dtype during the global-memory
store.

## Online-softmax recurrence

For an old state `(m_old, l_old, O_old)` and a new score tile:

```text
m_new = max(m_old, max(score_tile))
alpha = exp(m_old - m_new)
p     = exp(score_tile - m_new)

l_new = alpha * l_old + sum(p)
O_new = alpha * O_old + p @ V_tile
```

Only after all tiles have been processed is the output divided by `l_new`.
This is what allows the kernel to discard each score tile and avoid allocating
the complete `[sequence_length, sequence_length]` attention matrix.

## Launcher and dispatch (lines 170-223)

`flash_attention_launch<T, HEAD_DIM>` constructs the three-dimensional grid,
selects the specialized kernel, attaches the current PyTorch CUDA stream, and
launches it with `cudaLaunchKernelEx`.

`dispatch_flash_attention_head_dim<T>` converts the runtime head dimension into
one of four compile-time specializations:

```text
16, 32, 64, 128
```

The public `flash_attention` function (lines 268-296) validates the tensors,
selects the dtype with `AT_DISPATCH_FLOATING_TYPES_AND2`, and calls the
launcher.

## Current performance limitations

The implementation has the defining memory behavior of Flash Attention, but it
is still an educational baseline:

- Q/K dot products are serial within each lane.
- Q, K, and V copies are scalar rather than vectorized.
- QK and PV do not use tensor cores.
- K/V tiles are not double-buffered.
- Loads are synchronous rather than `cp.async` or TMA.
- There is no warp specialization between producers and consumers.
- Only forward self-attention is implemented; there is no backward pass,
  dropout, arbitrary mask, cross-attention, or grouped-query attention.

These are natural, independently measurable optimization steps for a future
implementation.
