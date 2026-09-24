# AI / Codex working instructions

Read `README.md`, `PROJECT_STATE.md`, and the official elimination task before changing the implementation.

## Canonical workflow / handoff discipline

**Codex is the primary coding agent and repository Markdown is the project memory.** The user should not have to repeat project context across chats.

Important repository status:
- `Jamoliddin-00i/WIUT-HACKATHON` is currently a **handoff / staging repository**, not the final canonical team repository.
- The canonical team repository is **`abdulhamid-n/saleh-traffic-events`** on GitHub. Jamoliddin is a collaborator.
- The canonical repo already contains teammate EDA under at least `reports/eda/C3905/` and `reports/eda/C3896/`, including charts, `eda.json`, and `direction_field.json` per video, plus scripts under `tools/eda/`.
- Hamid said C3897 and C3902 EDA will be added in the same structure.
- Raw YOLO11m track CSVs are too large for Git; Hamid can provide them separately if needed.
- When the user has the canonical repo locally, Codex should work there and stop treating the staging repo as the implementation target.

### Recommended local migration

Prefer a **fresh clone of the canonical repo** rather than repointing the staging repo's `origin`, because the repositories may have different histories.

Suggested Windows layout:

```text
D:\wiut hackathon\code\WIUT-HACKATHON       # staging / handoff repo
D:\wiut hackathon\code\saleh-traffic-events # canonical team repo
```

Clone command:

```powershell
cd "D:\wiut hackathon\code"
git clone https://github.com/abdulhamid-n/saleh-traffic-events.git
cd saleh-traffic-events
git pull origin main
```

Then open `D:\wiut hackathon\code\saleh-traffic-events` as the Codex workspace. Copy/cherry-pick only useful work from the staging repo after inspecting conflicts; do not blindly merge unrelated histories.

Before starting work in the canonical team repo:
1. Verify `git remote -v` and make sure `origin` points to `abdulhamid-n/saleh-traffic-events`.
2. `git pull` / sync `main`.
3. Read the canonical repo's own README/docs first.
4. Inspect `reports/eda/` and `tools/eda/` before recreating teammate work.
5. Carry over only the relevant project-state notes/scripts from the staging repo that are still needed.

After any meaningful discovery, benchmark, implementation decision, changed constraint, teammate handoff, or completed task:
- update `PROJECT_STATE.md` or the relevant Markdown doc in the same work session;
- if the information is already documented and still correct, **do not rewrite or duplicate it**;
- update an existing statement only when the fact actually changed;
- keep the current-state / next-actions section fresh so another Codex session can resume without chat history.

### Git authorship / contributors

The canonical team repository must show **human contributors only**.

- Commits made from the user's machine must use the user's configured Git identity.
- Do **not** add `Co-authored-by:` trailers for Codex, ChatGPT, Claude, or other AI assistants.
- Do **not** configure an AI/bot author identity.
- Do **not** create commits or PRs in the canonical team repo through an AI/bot GitHub identity when the same change can be made locally and committed by the user.
- AI tools may generate/edit code locally, but the visible Git author/contributor should remain the human team member who owns and submits the work.

## Hard requirement: models run locally

For development on the user's PC, **any AI/CV model used by this project must be installed/downloaded onto the PC and run locally on the user's GPU whenever CUDA is available.** Do not use hosted inference APIs for the traffic pipeline.

Examples:
- YOLO / RT-DETR weights: download to local `weights/` or the framework cache, then run with CUDA.
- Trackers and temporal models: local Python packages/code.
- Any fine-tuned weights: save locally and make the final submission able to load them offline.

Prefer GPU execution explicitly when supported (for example PyTorch `cuda`, Ultralytics `device=0`). Verify CUDA availability and report which GPU, CUDA/PyTorch versions, and VRAM are detected before heavy inference.

CPU fallback is acceptable only when a component does not benefit from GPU or CUDA is unavailable. Do not silently run a heavy detector on CPU if the GPU is usable.

Kaggle may be used for T4-like benchmarking or fine-tuning, but final inference must remain offline/reproducible from the submitted package.

## Final judging constraint

The judges run inference **offline** on an NVIDIA T4-class GPU with about 16 GB VRAM. Therefore development may use the user's GPU, but the final code must also fit and run reliably on that judge GPU, within the 3x-video-duration wall-clock limit and <=5 GB total model weights.

No OpenAI, Gemini, Anthropic, or other hosted model/API may be called during final inference. Coding assistants are allowed for development only.

### Decode/runtime warning

The official raw videos are 4K H.264 High 4:2:2 10-bit and the T4 cannot be assumed to hardware-decode that format. CPU decode is therefore part of the runtime budget. The organizer harness also calls `RiskEstimator.step()` on every frame after `cv2.VideoCapture` decodes/converts the frame to BGR, so frame decoding cannot simply be skipped for Part B.

Treat actual OpenCV decode benchmarking as a first-class performance constraint and keep Part A comfortably below the remaining budget. See `PROJECT_STATE.md` for current measurements and pending benchmark work.

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
