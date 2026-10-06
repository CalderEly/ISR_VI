# ISR_IV
Research Plan Versions: https://docs.google.com/document/d/1cpaLidQlgMXAjLGeAQVMJMn7QAF9FM-7QrrRme_I-cA/edit?usp=sharing 

Institustions for networking:
 1. USC
 2. MSOE

## TODO
### ISSUES

```diff
+ FIXED
- ISSUES PREVENTING RUN
! ISSUES THAT IMPEDE OUTPUT
# OPTIMIZATION TODOS
TODO
```





### Data Transfer (Fully formatted by AI)

  ### General rules
  
  - **Batched everything.** All methods take and return arrays/tensors. No per-neuron Python loops — operate on whole arrays at once. (Loops inside GPU kernels or I/O submission are fine.)
  - **Low latency.** Every method is on a hot path; research ways to cut latency (vectorized ops, batched I/O, avoiding small transfers).
  - **Tier constants:**
  
  | Constant | Value | Storage |
  |---|---|---|
  | `SSD` | 0 | Disk |
  | `RAM` | 1 | System memory |
  | `GPU` | 2 | VRAM |
  
  ---
  
  ## Methods
  
  ### `locate(ids) -> (tiers, addresses)`
  
  Looks up where each neuron lives.
  
  - **Input:** array of neuron IDs.
  - **Returns:** two arrays aligned with `ids` — `tiers[i]` and `addresses[i]` belong to `ids[i]`.
  - The tier map must be **readable from the GPU**, because Phase 3 calls this during spike delivery.
  
  ### `move(ids, dst_tier) -> handle`
  
  Moves neurons to another tier.
  
  - All `ids` must currently share **one source tier**; the caller groups them.
  - **Refuses or defers** any ID whose *in-simulation* flag is set.
  - Fails any IDs that don't fit in `dst_tier`.
  - A record moves **whole**: header, state, synapses, reverse list.
  - **Returns** a handle:
    - `handle.done() -> bool` — whether the move has finished
    - `handle.failed() -> ids` — IDs that did not move
  
  ### `capacity(tier) -> (used, free)`
  
  Reports how many neurons a tier holds and how many more fit.
  
  - Counted in neurons.
  - **Build this last** — it depends on record size, so the record format must be final first.
  
  ### `allocate(tier, n) -> (ids, addresses)`
  
  Reserves space for `n` new neurons.
  
  - Creates blank records: versioned header, flags cleared, empty synapse section, empty reverse list.
  - Assigns IDs and adds them to the tier map.
  - Updates the tier's used count.
  
  ### `free(ids) -> failed_ids`
  
  Releases neurons' space for reuse.
  
  - **Refuses** any ID whose *in-simulation* flag is set.
  - Removes IDs from the tier map, marks space reusable, updates the used count.
  - Does **not** clean up synapses on other neurons — Phase 3 does that before calling `free`.
  
  ---
  
  ## Record rules
  
  1. **Versioned header.** Every record carries a format version so new fields can be added without rewriting stored data.
  2. **Stable synapse slots.** A pruned synapse leaves an empty slot; slots never shift. Reverse-list entries point at (sender, slot) and would break if slots moved.
  3. **Synapse growth:** `[ fixed max per neuron | overflow blocks ]` — **decide before building `allocate`.**
  
  ---
  
  ## Open decisions
  
  - [ ] Synapse growth: fixed max or overflow blocks
  - [ ] Final record size (needed for `capacity`)

## DOCS

## Research

Items to research:
 1. Heterogeneuos neurons
 2. Structural plasticity
 3. Methods for measuring success

## PRESENT
