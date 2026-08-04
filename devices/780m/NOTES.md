# devices/780m — AMD Radeon 780M (gfx1103 APU) phased-memory variant

Goal: cut GPU memory by freeing the Qwen text-encoder weights after the context
tensor is computed, before the DiT phase begins. On the iGPU this turns the
long DiT phase from ~16 GB sustained into ~8-9 GB sustained.

Two sources in this folder, both with the same graph-split applied:
- `q5km.cpp` -> `zimage_q5km_780m` (Q5_K_M, mixed-quant async pipeline kept in
  the DiT phase).
- `q8.cpp`   -> `zimage_q8_780m`   (Q8_0; uniform quant, no async pipeline — the
  split is the same but simpler, no `zq8_begin_async_pipeline_capture`).

## Status: working and bit-exact at all three sizes

The graph-split is validated end-to-end. Same prompt/seed, phased
`zimage_q5km_780m` vs original `zimage_q5km`, on gfx1103 via
`HSA_OVERRIDE_GFX_VERSION=11.0.0`, GNOME stopped (TTY):

| size      | build   | exit | wall time | sustained DiT GTT | SHA-256                          |
| ----------| --------| ---- | --------- | ----------------- | -------------------------------- |
| 512x512   | original| 0    |  58.6 s   | ~16 GB            | `d196425d…0242013`               |
| 512x512   | phased  | 0    |  **51.5 s** | **~8.3 GB**     | `d196425d…0242013` ✅ identical |
| 576x1024  | original| 0    | 126.8 s   | ~16 GB            | `0f88ae34…88e5b`                 |
| 576x1024  | phased  | 0    | **121.9 s** | **~8.5 GB**     | `0f88ae34…88e5b`   ✅ identical |
| 1024x1024 | original| 0    | 257.4 s   | ~16 GB            | `f870fd22…4fea7`                 |
| 1024x1024 | phased  | 0    | **247.2 s** | **~8.8 GB**     | `f870fd22…4fea7`   ✅ identical |

Zero `[gfxhub] page fault`, zero `GPU reset begin` across all six runs
(no use-after-free of the freed Qwen arena).

### Q8_0 (zimage_q8_780m) — spot-checks

Same split applied to `q8.cpp`. Same prompt/seed, phased vs original:
```
512x512   f9ad36bdc35fe3b434a05ca44a64a76ebbc3b661a2cc67329e62e87696396cc8  phased  73,931 ms / orig  82,173 ms
576x1024  8ee896d8db5b6714937539d552d5b18aefaa6f1a63bf8278fca1f79a7c652d1e  phased 179,457 ms / orig 184,493 ms  (GTT peak 14.3 GB vs 17.1 GB)
```
Bit-identical at both sizes, zero faults/resets, phased ~3-10% faster. (Q8
1024x1024 is compile-clean, same split pattern, not yet runtime-tested.)

### Speed
Phased is faster at every size because lower sustained memory pressure means
fewer amdgpu `init_user_pages` pin retries:
- 512x512:   51.5 s vs 58.6 s  → **-12.1%**
- 576x1024: 121.9 s vs 126.8 s → **-3.9%**
- 1024x1024: 247.2 s vs 257.4 s → **-4.0%**

### Memory
- Sustained DiT-phase GTT is roughly halved (~16 GB → ~8-9 GB) — the win that
  matters, since DiT is the long phase.
- Transient peak at the phase-1→2 boundary (~16.4 GB at 1024x1024) momentarily
  matches the original's sustained level, but only for an instant; e.g. the
  1024x1024 signature:
  ```
  t=8s   7831 MiB   Qwen weights loaded
  t=16s 16447 MiB   phase-transition transient (Qwen + DiT + phase-2 workspace)
  t=24s+  8774 MiB   Qwen freed -> DiT steady-state for the ~4 min run
  ```

## How it works

A first attempt inserted a `hipStreamSynchronize` + `encoder_weights.free_early()`
right after `q22_enqueue_context_refiner`. That failed with
`hipErrorStreamCaptureUnsupported (900)`: the whole pipeline is captured into
ONE hip graph by `Q261CaptureGuard` (constructor calls `hipStreamBeginCapture`,
`finish()` calls `hipStreamEndCapture`), so a mid-capture sync is illegal.

The working approach splits the single captured graph into two, per `q22_run`
body (applied to all three sizes):

1. **Phase-1 capture (Qwen, BF16):** a nested `Q261CaptureGuard qwen_capture_guard`
   records `q22_enqueue_qwen_prompt`, `q22_enqueue_prompt_projection`,
   `q22_enqueue_context_refiner` (and their profiling events); `finish(qwen_graph)`;
   `hipGraphInstantiate` + `hipGraphUpload` + `hipGraphLaunch` + `hipStreamSynchronize`.
2. `encoder_weights.free_early()` — reclaims ~7.6 GB. The context output lives in
   its own `DeviceBuffer` (`context_workspace`, `prompt_projection.projected`) and
   persists across the launch boundary, so the DiT reads byte-identical context.
3. **Phase-2 capture (DiT + VAE):** the original `Q261CaptureGuard capture_guard`
   now starts here, immediately followed by `zq8_begin_async_pipeline_capture()`
   (the async quant-decode machinery stays with the DiT phase), the patchify +
   DiT step loop, VAE, `zq8_finalize_static_pipeline_capture()`, `finish(graph)`,
   `q20_staticize_quant_decode_graph`, instantiate/upload/launch/sync — unchanged.

### Supporting change
`zimage_qwen::DeviceWeightArena` gained a `free_early()` method
(`hipFree(base_); base_ = nullptr; bytes_ = 0;`). The destructor's `if (base_)`
guard makes this double-free safe. The sampler/vae arenas are untouched.

## Caveats

- `gpu_milliseconds` / "Full graph wall time" wraps only the phase-2 (DiT)
  launch; phase-1 (Qwen, ~3 s) is not included in that metric. Cosmetic — the
  table above was timed externally.
- Mode-1 (one-shot) only. `q22_run` is currently called only from mode-1
  `call_single` (3 call sites). Do NOT call `q22_run` from mode-2 (resident
  interactive) without reloading the Qwen arena — `encoder_weights` is freed at
  the end of phase 1 and the resident is reused across prompts in mode 2.
- Build target is gfx1100; run with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- GNOME must be stopped on the APU — its memory pressure causes amdgpu
  `init_user_pages: -1` floods and GPU resets. Drop to TTY first.

## Why the GGUF is staged on /dev/shm

The DiT weights (`--dit-gguf`) are mmap'd read-only (`PROT_READ, MAP_PRIVATE`,
file-backed) by `MappedFile`. On this APU the GGUF must be staged on tmpfs
(`/dev/shm`) rather than read from disk:

1. **File-backed page-pin failures (EPERM).** On kernel 6.12 + `amd_iommu=on`,
   amdgpu's `init_user_pages` intermittently returns `-EPERM` (-1) when pinning
   file-backed (ext4) pages for GPU access. From ext4 this floods — hundreds to
   thousands of `amdgpu: init_user_pages: Failed to get user pages: -1` — and
   eventually leaves a NULL GPU mapping that the shader faults on
   (`[gfxhub] page fault` at address 0x0) → MES wedge → MODE2 GPU reset.
   tmpfs pages are shmem-backed (anonymous-like) and pin cleanly, so the flood
   does not occur.
2. **Driver state accumulates across resets.** Each GPU reset strands GTT
   (observed ~16 GB unreclaimed after several resets), which then amplifies the
   pin failures on later runs. A reboot is required to clear it. tmpfs + TTY +
   a fresh boot is the reliable combo.

Disk-path workarounds that were tried and do **not** work on this 30 GB box:
- `PROT_READ|PROT_WRITE, MAP_PRIVATE` (give the VMA `VM_MAYWRITE`): suppresses
  the EPERM but the write-pin path on file-backed pages leaves a NULL GPU
  mapping → `[gfxhub] page fault` @0x0 → MES wedge.
- `MAP_ANONYMOUS` + `read()` into a private buffer: pins cleanly but
  double-buffers (page cache + anon copy) and OOMs the 30 GB box mid-load.

Note: the `--model-dir` Qwen/VAE safetensors read fine straight from disk
because they are read into host buffers and `hipMemcpy`'d — they are not pinned
as a file-backed userptr the way the GGUF mmap is.

## Run

```bash
sudo systemctl isolate multi-user.target        # stop GNOME
export HSA_OVERRIDE_GFX_VERSION=11.0.0
./zimage_q5km_780m 1 --size 512x512 \           # or 576x1024 / 1024x1024
  --model-dir models/z_image_turbo_ms \
  --dit-gguf /dev/shm/<gguf staged on tmpfs> \
  --prompt "..." --seed 42 --max-steps 8 --output out.ppm
```
