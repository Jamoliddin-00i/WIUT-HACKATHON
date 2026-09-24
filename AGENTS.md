# AI / Codex working instructions

Read `README.md` and the official elimination task before changing the implementation.

## Hard requirement: models run locally

For development on the user's PC, **any AI/CV model used by this project must be installed/downloaded onto the PC and run locally on the user's GPU whenever CUDA is available.** Do not use hosted inference APIs for the traffic pipeline.

Examples:
- YOLO / RT-DETR weights: download to local `weights/` or the framework cache, then run with CUDA.
- Trackers and temporal models: local Python packages/code.
- Any fine-tuned weights: save locally and make the final submission able to load them offline.

Prefer GPU execution explicitly when supported (for example PyTorch `cuda`, Ultralytics `device=0`). Verify CUDA availability and report which GPU, CUDA/PyTorch versions, and VRAM are detected before heavy inference.

CPU fallback is acceptable only when a component does not benefit from GPU or CUDA is unavailable. Do not silently run a heavy detector on CPU if the GPU is usable.

## Final judging constraint

The judges run inference **offline** on an NVIDIA T4-class GPU with about 16 GB VRAM. Therefore development may use the user's GPU, but the final code must also fit and run reliably on that judge GPU, within the 3x-video-duration wall-clock limit and <=5 GB total model weights.

No OpenAI, Gemini, Anthropic, or other hosted model/API may be called during final inference. Coding assistants are allowed for development only.

## Local assets

The huge official MP4 samples stay local and must never be committed. Find them in the user's local video folder and ensure `.gitignore` covers them.

## Official task PDF

Expected local path after bootstrapping:

`docs/WIUT Hackathon _ CV Track Elimination Task.pdf`

If it is missing, run:

```bash
python scripts/fetch_official_task.py
```

The source is the organizer-provided Google Drive file. After it is downloaded on the user's PC, it may be committed because it is only ~254 KB.

## Preserve organizer files

Do not modify `run_submission.py` or `evaluate.py`. Keep `solution.py` compatible with the exact starter-kit interface.
