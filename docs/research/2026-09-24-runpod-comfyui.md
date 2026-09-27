# Is RunPod a viable alternative to the local ComfyUI (RTX A5000, 24 GB) for this project's GPU workload?

## Short answer

No, not at the measured usage level: the local RTX A5000 already fits the 24 GB floor, costs nothing marginal, and the 58 GiB weight set is the deciding factor — RunPod's network-volume read throughput (200–400 MB/s) means a cold worker would spend roughly 2.5–5 minutes just streaming those weights before generation starts, on top of RunPod's own cold-start overhead, for a workload whose "cell" is already 85 s wall-clock. RunPod becomes clearly worth it only for scenarios the A5000 structurally cannot do: needing >24 GB VRAM, running several matrix cells in true parallel, or freeing the workstation during long batch runs — none of which apply to the current 32–64-cell target. The unauthenticated ComfyUI HTTP API is also a real problem on RunPod: exposing a Pod's HTTP port gives you a guessable-but-public proxy URL with no built-in auth, so the client's current "trust `COMFYUI_URL`" model would need an added auth layer before it could safely leave the LAN.

## Findings

### 1. Pods vs Serverless — which fits a long interactive session with idle gaps

- **Pods** are billed by the second while running, for both compute and storage, with "no fees for data ingress or egress"; GPU billing stops the instant a Pod is stopped, but a Pod's **Volume Disk** keeps accruing $0.20/GB-month even while stopped, and a **Network Volume** keeps accruing $0.07–$0.05/GB-month whether attached to a running or stopped Pod. [docs.runpod.io/pods/pricing](https://docs.runpod.io/pods/pricing) (fetched 2026-09-24)
- A Pod persists a real, addressable HTTP endpoint for as long as it's running (via the `https://<pod-id>-<port>.proxy.runpod.net` proxy, see §5) — this is the shape that matches "generate, look, adjust, regenerate" sessions, at the cost of paying for idle GPU time between bursts unless you manually stop/restart the Pod each time. [docs.runpod.io/pods/configuration/expose-ports](https://docs.runpod.io/pods/configuration/expose-ports) (fetched 2026-09-24)
- **Serverless** is built for bursty, stateless request/response traffic: with `activeWorkers = 0` you pay nothing while idle and only for the seconds a worker is actually processing a request, but every idle-to-active transition is a cold start (weights reload, see §4), and an endpoint idle for 7 days has its max workers auto-set to 0 by RunPod (any request resets the clock). Setting `activeWorkers ≥ 1` removes cold starts but bills that worker continuously, defeating the "idle gaps cost nothing" property. [docs.runpod.io/serverless/endpoints/endpoint-configurations](https://docs.runpod.io/serverless/endpoints/endpoint-configurations) (fetched 2026-09-24)
- A persistent, addressable ComfyUI HTTP endpoint (matching the client's single `COMFYUI_URL` env var, polling `/history` etc.) is practical on **Pods** (the URL stays stable while the Pod runs) but not really on Serverless in its natural mode — Serverless's request/response job model (`POST` a job, poll `/status` or `/stream`) doesn't map cleanly onto ComfyUI's own native HTTP API (`/prompt`, `/history`, `/view`) without wrapping it in a RunPod handler function, which is extra integration work this project's client does not currently do. [docs.runpod.io/serverless/workers/handler-functions](https://docs.runpod.io/serverless/workers/handler-functions) (fetched 2026-09-24)

### 2. Current GPU pricing, ≥24 GB VRAM, as of 2026-09-24

Community Cloud and Secure Cloud are still two separate tiers for Pods, per RunPod's own pricing page (fetched 2026-09-24):

| GPU | VRAM | Community Cloud (Pods) | Secure Cloud (Pods) | Serverless |
|---|---|---|---|---|
| RTX A5000 | 24 GB | $0.16/hr | $0.27/hr | $0.69/hr (grouped w/ L4, 3090, MIG-24GB) |
| RTX 4090 | 24 GB | $0.34/hr | $0.74/hr | (grouped above) |
| RTX 3090 | 24 GB | $0.22/hr | $0.50/hr | (grouped above) |
| L4 | 24 GB | $0.44/hr | $0.49/hr | (grouped above) |
| RTX 5090 | 32 GB | $0.69/hr | $0.99/hr | $1.58/hr |
| A6000 / A40 | 48 GB | $0.33–0.35/hr | $0.49–0.53/hr | $1.22/hr |
| L40 / L40S / 6000 Ada | 48 GB | $0.69–0.79/hr | $0.82–1.09/hr | $1.75/hr |
| A100 (80 GB) | 80 GB | $1.19–1.39/hr | $1.59/hr | $2.72/hr |
| H100 | 80–94 GB | $1.99–2.59/hr | $2.89–3.49/hr | $4.79/hr |
| H200 | 141 GB | $3.59/hr | $4.59/hr | $5.93/hr |

[runpod.io/pricing](https://www.runpod.io/pricing) (fetched 2026-09-24)

- The workload's floor (24 GB) is met even at the cheapest tier, so RunPod offers no VRAM headroom the A5000 doesn't already have unless you deliberately step up to a 32 GB+ card — relevant only if the eventual full matrix (≈double the near-term 32-cell target) needs to run multiple models resident simultaneously.
- Serverless is consistently ~2–4× the matching Pod Secure Cloud rate per GPU-hour, which is the premium for scale-to-zero + auto-scaling infrastructure; for a single-user interactive workload that premium buys mostly cold-start pain (§4), not benefit.
- RunPod states Secure Cloud runs on "T3/T4 data centers" with high redundancy, while Community Cloud is a peer-to-peer host network with "variable" reliability; RunPod's docs also note they are "no longer accepting new hosts for Community Cloud," suggesting a strategic wind-down of that cheaper tier. [docs.runpod.io/pods/choose-a-pod](https://docs.runpod.io/pods/choose-a-pod) (fetched 2026-09-24)

### 3. The 58 GiB weight problem — network volumes

- Standard network volume storage: **$0.07/GB-month for the first 1 TB**, then $0.05/GB-month beyond that; a High-Performance tier exists at a premium with up to 3× the throughput. 58 GiB ≈ 62.3 GB → **≈$4.36/month** to keep the full model set resident on a standard volume, whether or not any Pod using it is running. [docs.runpod.io/storage/network-volumes](https://docs.runpod.io/storage/network-volumes) (fetched 2026-09-24)
- Standard network volumes are rated at **200–400 MB/s sustained, up to 10 GB/s peak**. At sustained throughput, reading the full 58 GiB (≈62.3 GB) model set cold takes **≈156–312 s** (2.6–5.2 min) — comparable to or worse than the local cold-load measurement of 154–173 s, and that's before RunPod's own container/infra cold-start overhead is added on top. [docs.runpod.io/storage/network-volumes](https://docs.runpod.io/storage/network-volumes) (fetched 2026-09-24)
- Data on a network volume "is retained when Pods terminate or Serverless workers scale to zero" — it persists across restarts, unlike a Pod's ephemeral Container Disk. [docs.runpod.io/storage/network-volumes](https://docs.runpod.io/storage/network-volumes) (fetched 2026-09-24)
- Attaching a network volume to a Pod is a one-time deployment decision — it "cannot be attached or detached later without deleting the Pod." For Serverless, a volume mounts at `/runpod-volume` in every worker but pins the endpoint's workers to that volume's datacenter, which can reduce GPU availability. [docs.runpod.io/storage/network-volumes](https://docs.runpod.io/storage/network-volumes) (fetched 2026-09-24)
- A third-party benchmark (not RunPod's own figures — flagged as secondary) measured Llama-3-70B-Instruct cold-start on an A40, over ~4,200 serverless invocations across 10 days in June 2026: **248 s** loading from a fresh public download vs **42 s** loading from a pre-populated Network Volume (an ~83% reduction) — this supports the direction (network volumes help) but the project's own weight set is smaller (58 GiB vs Llama-3-70B's ~140 GB at fp16), so the absolute seconds don't transfer directly; use the §3 sustained-throughput math above as the primary-source-grounded estimate instead. [markaicode.com/errors/runpod-timeout-fix](https://markaicode.com/errors/runpod-timeout-fix/), published 2026-06-25 (third-party, secondary source)
- Net read: the 58 GiB model set is squarely in the range where a network volume "solves" the redownload problem but does not make it free — every cold start (Pod boot or Serverless scale-from-zero) still pays a multi-minute tax to stream the weights into GPU/host memory, on a workload where the local baseline already pays that tax once per session (154–173 s) and then serves it warm.

### 4. Cold start / spin-up latency

- RunPod's official guidance splits Serverless cold start into "infrastructure cold start" (provisioning compute, attaching storage, pulling the container image) and application-level cold start (loading weights, `torch.compile`, CUDA graph capture) — both are on the critical path for every scale-from-zero event. [runpod.io/blog/cut-vllm-cold-starts-runpod-serverless](https://www.runpod.io/blog/cut-vllm-cold-starts-runpod-serverless), updated 2026-09-13
- RunPod's own case study (same source) reduced a 32B FP8 vLLM model's cold start from 324 s to 91 s on 2×H200 through configuration tuning — i.e. even after optimization, RunPod's own numbers show ~90 s of unavoidable cold-start overhead for a model roughly the size of this project's combined weight set, before any generation happens. [runpod.io/blog/cut-vllm-cold-starts-runpod-serverless](https://www.runpod.io/blog/cut-vllm-cold-starts-runpod-serverless), updated 2026-09-13
- **FlashBoot** is RunPod's Serverless feature (enabled by default) that "retains worker state after spin-down" for faster revival than a fresh boot — it helps warm-to-idle-to-warm cycling within a live traffic window, but does not eliminate the first cold start of a session, which is the relevant case for this project's bursty, gap-heavy usage pattern. [docs.runpod.io/serverless/endpoints/endpoint-configurations](https://docs.runpod.io/serverless/endpoints/endpoint-configurations) (fetched 2026-09-24)
- Net: even in the best case, a Serverless cold start (infra + weight load, network-volume-backed) for this project's model set plausibly lands in the same 2–5 minute range as the local cold-load baseline (154–173 s) — RunPod does not clearly beat the number this project already measures locally, and every idle-gap-then-resume cycle in the bursty usage pattern described would re-pay that cost, whereas the local machine pays it once per session.

### 5. Exposing ComfyUI safely

- RunPod's own Pod networking docs state plainly: **"Runpod does not provide built-in authentication"** for HTTP-proxied ports. The proxy URL format is `https://<pod-id>-<port>.proxy.runpod.net`; RunPod's docs call the Pod ID "some obscurity" but explicitly instruct: "implement proper authentication in your application." [docs.runpod.io/pods/configuration/expose-ports](https://docs.runpod.io/pods/configuration/expose-ports) (fetched 2026-09-24)
- The alternative — **Expose TCP Ports** — gives direct TCP forwarding to a public IP, again with no RunPod-managed auth. [docs.runpod.io/pods/configuration/expose-ports](https://docs.runpod.io/pods/configuration/expose-ports) (fetched 2026-09-24)
- RunPod does offer **"global networking"** for private Pod-to-Pod communication without exposing ports publicly, but this only helps if every client of ComfyUI is itself another RunPod Pod — it does not cover a client running on the user's own Fedora workstation. [docs.runpod.io/pods/networking](https://docs.runpod.io/pods/networking) (fetched 2026-09-24) — page content on the private-networking mechanics itself is thin; flagged as not fully verified below.
- Conclusion for this project: the client currently sends **zero** authentication headers to `COMFYUI_URL` (per the task brief). Pointing it at an unmodified RunPod Pod proxy URL means the world can reach that ComfyUI instance (upload images, submit prompts, burn the user's GPU-hour balance) unless the client or a reverse proxy in front of it adds auth first. This is not avoidable through RunPod configuration alone — it requires an application-level change (e.g. a bearer token check, or tunnelling through something like a Cloudflare Access / WireGuard layer) before it would be safe to leave running unattended.

### 6. Bandwidth / egress costs

- RunPod states plainly: "no fees for data ingress or egress" on Pods. Search aggregation of RunPod's pricing (secondary summaries, cross-checked against the same "no ingress/egress fees" language on RunPod's own pricing page) confirms Secure Cloud has zero egress fees; some **Community Cloud** hosts may pass through their own network costs since that tier is peer-to-peer infrastructure, but this is not itemized by RunPod centrally. [docs.runpod.io/pods/pricing](https://docs.runpod.io/pods/pricing) (fetched 2026-09-24); Community Cloud caveat per secondary aggregation, not independently confirmed against a primary RunPod statement — flagged below.
- For this project's traffic pattern (PNG/short-mp4 uploads, ~21 PNGs at 512×768 pulled back per cell, `/history` polling every 2 s), bandwidth cost itself is not the concern — those payloads are small (a 512×768 PNG is typically well under 1 MB) and egress is free on Secure Cloud regardless of volume.

### 7. Maintained ComfyUI template — pinning models, custom nodes

- RunPod maintains an official ComfyUI Pod template ("built from the comfyui-base repository"), with a Blackwell-specific variant for RTX 5090/B200, pre-loaded with ComfyUI-Manager, KJNodes, Civicomfy, and ComfyUI-RunpodDirect. [docs.runpod.io/tutorials/pods/comfyui](https://docs.runpod.io/tutorials/pods/comfyui) (fetched 2026-09-24)
- The official template does **not** ship models pre-installed — models are added post-deploy via ComfyUI-Manager or manual download, which means this project's `gpu/models.lock`-pinned, sha256-verified fetch process would need to be re-run (or baked into a custom Docker image) on top of the base template rather than relying on anything RunPod pins for you. [docs.runpod.io/tutorials/pods/comfyui](https://docs.runpod.io/tutorials/pods/comfyui) (fetched 2026-09-24)
- The template's documentation does not address custom-node installation specifics (Nunchaku, the Z-Image ControlNet node) directly, but since a Pod is a full VM with shell access, both are installable exactly as on the local machine: via ComfyUI-Manager's custom-node search, or a manual `git clone` into `custom_nodes/` per Nunchaku's own install docs. [nunchaku.tech/docs/ComfyUI-nunchaku/get_started/installation.html](https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/installation.html) (secondary confirmation of general install path, fetched via search 2026-09-24) — this is standard ComfyUI custom-node installation, not a RunPod-specific mechanism, so it works but gives no advantage over the current local setup; it would need to be re-scripted for a Pod image rather than reused as-is.
- Net: nothing about RunPod's template removes the need to reproduce this project's own `gpu/models.lock` pinning and Nunchaku/ControlNet custom-node setup — it would need to be baked into a custom Docker image (recommended by RunPod's own optimization docs for private/pinned models, see §4) to avoid re-running the whole download-and-verify process on every fresh Pod.

### 8. Latency realism — polling `/history` every 2 s, pulling ~21 PNGs per cell, over a WAN

- Nothing found in RunPod's documentation suggests HTTP polling or file downloads at this scale are impractical over their network; the constraint is not bandwidth or protocol but the **added round-trip latency and jitter of a WAN link** versus the current LAN/localhost setup, plus the cold-start tax on the far end (§4) that a 2-second poll loop would simply wait through. RunPod's own docs do not discuss client-side polling patterns at all — this specific point is a reasoning inference from the general WAN-vs-LAN latency difference and the cold-start figures already sourced above, not a directly documented claim; flagged as inferred, not verified against a primary source.
- Concretely: 21 PNGs × maybe 200–500 KB each at 512×768 is a few MB total — trivial over any broadband connection; the operationally relevant cost is the wall-clock gap a human notices between "click generate" and "see result," which is dominated by cold-start (minutes) and Wan inference time (9–20 s per clip, per the project's own warm-GPU measurements) rather than by network transfer of the output PNGs.

### SAM 2.1 segmentation (CPU-only step, ~10–45 s per job)

- Nothing found in RunPod's product surface changes the brief's own assessment. Moving a ~10–45 s CPU job to a remote GPU pod adds: a cold-start tax (§4, likely 90 s+ even optimized) if the pod isn't already warm, or a continuously-billed idle GPU (§1) if it is — either way the round-trip cost dwarfs the local CPU time, and SAM 2.1 doesn't benefit from a GPU enough (it's already listed as CPU-bound in this project) to justify holding a Pod or Serverless worker warm just for it. RunPod offers CPU-only Pods as a separate product line, but that's equivalent to what already runs locally, at nonzero marginal cost, with no latency win over the workstation's own CPU. [docs.runpod.io/pods/overview](https://docs.runpod.io/pods/overview) (fetched 2026-09-24, general product description; CPU-Pod-specific pricing not independently pulled — low priority since the conclusion doesn't hinge on the exact number)

## Recommendation

**Stay on the local RTX A5000 for the near-term 32-cell target and the eventual ~64-cell full matrix.** The baseline ("already works, costs nothing marginal") is not beaten by RunPod at this scale:

- **VRAM**: the workload fits in 24 GB today; RunPod offers no VRAM RunPod can't already provide locally.
- **Cold weight-load tax**: RunPod's own numbers (§3, §4) put a network-volume-backed cold start for a comparable-sized model in the same 90 s–5 min range as the local machine's already-measured 154–173 s cold load — and the bursty, gap-heavy usage pattern means this tax would be paid repeatedly (every time a Pod is stopped to save money, or a Serverless worker scales to zero), where the local machine pays it once per session and then stays warm.
- **Auth**: the client sends no authentication at all; putting it on a RunPod Pod's public proxy URL unmodified is a real, avoidable-only-by-code-change security gap that doesn't exist when everything stays on localhost/LAN.
- **Cost**: even the cheapest matching Pod (RTX A5000, Community Cloud, $0.16/hr) times the near-term 40–50 min of GPU-card-time target is only ≈$0.11–$0.13 in raw compute — trivially cheap — but that's not the real cost; the real cost is the operational overhead of managing weight sync, cold starts, and auth for a marginal-dollar saving over a machine that's already paid for and sitting idle otherwise.

**RunPod would clearly be the right call if any of these become true:**
- The workload needs **>24 GB VRAM** resident at once (e.g. running Wan 2.2 Fun-Control and Qwen-Image-Edit simultaneously without unloading, or a bigger future model) — then even the cheapest 48 GB tier (A40/A6000, $0.33–0.53/hr) beats buying new hardware.
- The user wants **true parallel matrix cells** — N Pods running concurrently to cut wall-clock time on the full matrix, which the single local A5000 cannot do at all.
- The user wants the **workstation free** for other GPU work (e.g. gaming, another project) during a long batch run, and is willing to pay the small hourly rate for that.

**RunPod is clearly the wrong call for:** the current interactive, single-GPU, 24-GB-fits, bursty-with-gaps workload as measured — the marginal cost saved on hardware (which is $0, already owned) is smaller than the new operational cost (weight sync to a network volume, cold-start latency repeated per session, and building an auth layer that doesn't currently exist).

## Sources

- https://www.runpod.io/pricing (fetched 2026-09-24)
- https://docs.runpod.io/pods/overview (fetched 2026-09-24)
- https://docs.runpod.io/pods/pricing (fetched 2026-09-24)
- https://docs.runpod.io/pods/choose-a-pod (fetched 2026-09-24)
- https://docs.runpod.io/pods/networking (fetched 2026-09-24)
- https://docs.runpod.io/pods/configuration/expose-ports (fetched 2026-09-24)
- https://docs.runpod.io/storage/network-volumes (fetched 2026-09-24)
- https://docs.runpod.io/serverless/storage/network-volumes (fetched 2026-09-24)
- https://docs.runpod.io/serverless/endpoints/endpoint-configurations (fetched 2026-09-24)
- https://docs.runpod.io/serverless/workers/handler-functions (fetched 2026-09-24)
- https://docs.runpod.io/serverless/development/optimization (fetched 2026-09-24)
- https://docs.runpod.io/tutorials/pods/comfyui (fetched 2026-09-24)
- https://www.runpod.io/blog/cut-vllm-cold-starts-runpod-serverless (updated 2026-09-13, fetched 2026-09-24)
- https://nunchaku.tech/docs/ComfyUI-nunchaku/get_started/installation.html (fetched via search 2026-09-24)
- https://markaicode.com/errors/runpod-timeout-fix/ (third-party benchmark, published 2026-06-25, fetched 2026-09-24)

## Blocked

- https://www.answeroverflow.com/m/1393957012555038771 (Discord thread on "How do people serve large models on Runpod Serverless?") · 429 rate-limited · retry later or fetch manually; was being used only to cross-check the "60 GB streaming, ~2 min, costs more than inference" community anecdote, which was not independently confirmed and is omitted from Findings as a result
- RunPod's private/global-networking mechanics (Pod-to-Pod without public ports) · docs page content was thin on specifics beyond a one-line description · would need docs.runpod.io/pods/networking's linked sub-pages fetched individually, or a RunPod support/sales conversation, to confirm whether it could ever expose ComfyUI to a non-RunPod client (e.g. via a WireGuard bridge) without a public proxy URL
- Exact CPU-only Pod pricing (for the SAM 2.1 cross-check) · not independently fetched · low priority, since the conclusion (network round-trip exceeds the CPU-side saving) doesn't depend on the exact hourly figure; could pull from runpod.io/pricing's CPU tab if needed
