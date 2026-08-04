# devices/780m — AMD Radeon 780M (gfx1103 APU) phased-memory variant

Goal: cut peak GPU memory by freeing the Qwen text-encoder weights after the
context tensor is computed, before the DiT phase begins. Peak drops from
~29 GB (Qwen 7.6 + DiT 5.2 + workspace ~16) to ~21 GB; observed DiT-phase GTT
~8.3 GB vs ~16 GB for the original.

## Status: working (graph split)

A first attempt inserted a `hipStreamSynchronize` + `encoder_weights.free_early()`
right after `q22_enqueue_context_refiner`. That failed with
`hipErrorStreamCaptureUnsupported (900)` because the whole pipeline is captured
into ONE hip graph by `Q261CaptureGuard` (constructor calls
`hipStreamBeginCapture`, `finish()` calls `hipStreamEndCapture`). A mid-capture
sync is illegal.

The working approach splits the single captured graph into two, per `q22_run`
body (512x512 validated bit-exact; 576x1024 / 1024x1024 implemented + compile
clean, not yet runtime-tested):

1. **Phase-1 capture (Qwen, BF16):** a nested `Q261CaptureGuard qwen_capture_guard`
   records `qwen_enqueue_qwen_prompt`, `q22_enqueue_prompt_projection`,
   `q22_enqueue_context_refiner` (and their profiling events); `finish(qwen_graph)`;
   `hipGraphInstantiate` + `hipGraphUpload` + `hipGraphLaunch` + `hipStreamSynchronize`.
2. `encoder_weights.free_early()` — reclaims ~7.6 GB. The context output lives in
   its own `DeviceBuffer` (`context_workspace`, `prompt_projection.projected`) and
   persists across the launch boundary.
3. **Phase-2 capture (DiT + VAE):** the original `Q261CaptureGuard capture_guard`
   now starts here, immediately followed by `zq8_begin_async_pipeline_capture()`
   (the async quant-decode machinery stays with the DiT phase), the patchify + DiT
   step loop, VAE, `zq8_finalize_static_pipeline_capture()`, `finish(graph)`,
   `q20_staticize_quant_decode_graph`, instantiate/upload/launch/sync — unchanged.

### Supporting change
`zimage_qwen::DeviceWeightArena` gained a `free_early()` method
(`hipFree(base_); base_ = nullptr; bytes_ = 0;`). The destructor's
`if (base_)` guard makes this double-free safe. The sampler/vae arenas are
untouched.

## Validation (512x512, gfx1103 via HSA_OVERRIDE, TTY)

Same prompt/seed, phased vs original `zimage_q5km`:
```
d196425d5e47145f074f3a3cf160fccfa8eea2c99b47728bfa97ba2530242013  phased   51,526 ms
d196425d5e47145f074f3a3cf160fccfa8eea2c99b47728bfa97ba2530242013  original 58,624 ms
```
Bit-identical PPM, zero gfxhub page faults (no use-after-free), zero GPU resets.
Phased is ~12% faster (less memory pressure -> fewer amdgpu pin retries).

## Known caveats

- `gpu_milliseconds` / "Full graph wall time" still wraps only the phase-2
  (DiT) launch; phase-1 (Qwen, ~3 s) is not included in that metric. Cosmetic.
- The phased free is safe for mode-1 (one-shot) only. `q22_run` is currently
  called only from mode-1 `call_single` (3 call sites); do NOT call `q22_run`
  from mode-2 (resident interactive) without reloading the Qwen arena, since
  `encoder_weights` is freed at the end of phase 1 and the resident is reused
  across prompts in mode 2.
- Build target is still gfx1100; run with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- 576x1024 / 1024x1024 splits are compile-verified only — run-test before
  relying on them.

## Run (512x512, TTY)

GNOME must be stopped (its memory pressure causes amdgpu `init_user_pages: -1`
floods and GPU resets):

```bash
sudo systemctl isolate multi-user.target
export HSA_OVERRIDE_GFX_VERSION=11.0.0
./zimage_q5km_780m 1 --size 512x512 \
  --model-dir models/z_image_turbo_ms \
  --dit-gguf /dev/shm/<gguf staged on tmpfs> \
  --prompt "..." --seed 42 --max-steps 8 --output out.ppm
```
