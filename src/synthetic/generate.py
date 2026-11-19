"""#6/M3: synthetic condition images + manifest.

Two generators:
  procedural  (default, offline): class-distinct images drawn with PIL. Clearly labelled
              placeholders (generator="procedural-v1"); they exercise the manifest, the
              automated check, the demo and the Set S plumbing with no downloads.
  sdxl        (gated): real photoreal generation via diffusers on Apple MPS, using the
              per-class prompts below. Different model family from the readers (Hard rule #6).

Real VLM-on-photo numbers require the sdxl path (or supplied photos) + a reader key/Ollama.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

IMG_DIR = Path("data/synthetic/images")
MANIFEST = Path("data/synthetic/manifest.csv")
SIZE = 256

PITCH_CLASSES = ["green", "dry", "flat", "slow", "wet"]
SKY_CLASSES = ["clear", "overcast", "rain_now", "night"]
UNINFORMATIVE = "uninformative"

# Strong per-class prompts for the real image generators (SDXL / Gemini). Stored verbatim in the
# manifest as the prompt actually used, so the generation is reproducible. No team/venue/date text
# (memorisation control, Hard rule #7); "no text/people" keeps the pitch condition unobstructed.
_STYLE = "photorealistic, high detail, natural daylight, no text, no watermark, no logos"
PROMPTS = {
    "green": f"extreme close-up photograph of a fresh green grassy cricket pitch, moist surface with a layer of live grass, overcast morning light, 35mm, {_STYLE}, no people",
    "dry": f"extreme close-up photograph of a dry cracked brown cricket pitch, deep visible cracks in sun-baked clay, bare of grass, harsh midday sun, 35mm, {_STYLE}, no people",
    "flat": f"extreme close-up photograph of a hard flat even light-brown cricket pitch, well-rolled smooth clay, clear day, 35mm, {_STYLE}, no people",
    "slow": f"extreme close-up photograph of a dusty worn slow cricket pitch, powdery grey-brown dust and scuff marks, subcontinental ground, 35mm, {_STYLE}, no people",
    "wet": f"photograph of a cricket pitch under flat protective covers with a wet waterlogged outfield and puddles during rain, {_STYLE}",
    "clear": f"photograph of a clear blue sky over a cricket stadium, bright sunshine, few clouds, {_STYLE}",
    "overcast": f"photograph of a heavy grey overcast sky over a cricket ground, thick low clouds, flat diffuse light, {_STYLE}",
    "rain_now": f"photograph of steady rain falling over a cricket ground, dark wet sky, visible drizzle and puddles, {_STYLE}",
    "night": f"photograph of a floodlit cricket stadium at night, bright floodlight towers, dark sky, {_STYLE}",
    "uninformative": f"wide blurry out-of-focus photograph of a packed cricket stadium crowd, motion blur, {_STYLE}",
}

BASE_COLORS = {
    "green": (60, 120, 50), "dry": (170, 120, 70), "flat": (190, 165, 120),
    "slow": (150, 135, 110), "wet": (70, 70, 80),
    "clear": (90, 150, 230), "overcast": (150, 150, 155), "rain_now": (90, 95, 110),
    "night": (20, 20, 40), "uninformative": (120, 120, 120),
}


def _noise(draw, rng, color, n=1200, spread=25):
    for _ in range(n):
        x, y = rng.randint(0, SIZE - 1), rng.randint(0, SIZE - 1)
        c = tuple(max(0, min(255, v + rng.randint(-spread, spread))) for v in color)
        draw.point((x, y), fill=c)


def _render(cls: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    base = BASE_COLORS[cls]
    img = Image.new("RGB", (SIZE, SIZE), base)
    d = ImageDraw.Draw(img)
    if cls in ("clear", "overcast", "rain_now", "night"):
        # vertical sky gradient
        for y in range(SIZE):
            f = y / SIZE
            c = tuple(int(base[i] * (1 - 0.3 * f) + 255 * 0.1 * f) for i in range(3))
            d.line([(0, y), (SIZE, y)], fill=c)
        if cls == "rain_now":
            for _ in range(300):
                x = rng.randint(0, SIZE); y = rng.randint(0, SIZE)
                d.line([(x, y), (x - 3, y + 10)], fill=(200, 200, 210))
        if cls == "night":
            for _ in range(6):
                x = rng.randint(20, SIZE - 20)
                d.ellipse([x - 12, 10, x + 12, 34], fill=(255, 255, 210))
    else:
        _noise(d, rng, base, n=2500, spread=22)
        if cls == "dry":
            for _ in range(10):  # cracks
                x, y = rng.randint(0, SIZE), rng.randint(0, SIZE)
                pts = [(x, y)]
                for _ in range(6):
                    x += rng.randint(-18, 18); y += rng.randint(-18, 18)
                    pts.append((x, y))
                d.line(pts, fill=(90, 60, 30), width=1)
        if cls == "green":
            for _ in range(2000):  # grass blades
                x, y = rng.randint(0, SIZE), rng.randint(0, SIZE)
                d.line([(x, y), (x, y - rng.randint(1, 4))], fill=(40, 100, 40))
        if cls == "wet":
            for _ in range(12):
                x, y = rng.randint(0, SIZE), rng.randint(0, SIZE)
                r = rng.randint(6, 20)
                d.ellipse([x, y, x + r, y + r // 2], fill=(120, 130, 150))
    if cls == UNINFORMATIVE:
        img = img.filter(ImageFilter.GaussianBlur(4))
        d = ImageDraw.Draw(img)
        _noise(d, rng, base, n=3000, spread=60)
    return img


def _plan(per_class: int, n_uninformative: int, seed: int):
    """Shared (id, file, class, kind, seed) plan so every generator writes an identical manifest."""
    items = []
    sid = 0
    for cls, kind in [(c, "pitch") for c in PITCH_CLASSES] + [(c, "sky") for c in SKY_CLASSES]:
        for i in range(per_class):
            sid += 1
            name = f"{kind}_{cls}_{i:02d}.png"
            items.append((name[:-4], name, cls, kind, seed * 1000 + sid))
    for i in range(n_uninformative):
        sid += 1
        name = f"uninformative_{i:02d}.png"
        items.append((name[:-4], name, UNINFORMATIVE, "other", seed * 1000 + sid))
    return items


def _run(make_image, generator: str, out_dir: Path, per_class: int,
         n_uninformative: int, seed: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for iid, name, cls, kind, s in _plan(per_class, n_uninformative, seed):
        img = make_image(cls, s)
        img.save(out_dir / name)
        manifest.append({"id": iid, "file": name, "class": cls, "kind": kind,
                         "prompt": PROMPTS[cls], "generator": generator, "seed": s})
    _write_manifest(manifest)
    return manifest


def generate_procedural(out_dir: Path = IMG_DIR, per_class: int = 10,
                        n_uninformative: int = 20, seed: int = 0) -> list[dict]:
    """Offline class-distinct PIL placeholders (labelled generator='procedural-v1')."""
    return _run(_render, "procedural-v1", out_dir, per_class, n_uninformative, seed)


def _sdxl_turbo_pipe():  # pragma: no cover - needs torch/diffusers + MPS
    from diffusers import AutoPipelineForText2Image  # type: ignore
    import torch  # type: ignore

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    ).to("mps")
    pipe.set_progress_bar_config(disable=True)
    return pipe, torch


def generate_sdxl(out_dir: Path = IMG_DIR, per_class: int = 10, n_uninformative: int = 20,
                  seed: int = 0, steps: int = 4) -> list[dict]:  # pragma: no cover
    """Real photoreal generation via SDXL-Turbo on Apple MPS (fast: ~2-5s/image, guidance off).

    Stability family != the reader families (Claude/Qwen), satisfying Hard rule #6.
    """
    pipe, torch = _sdxl_turbo_pipe()

    def make_image(cls: str, s: int):
        g = torch.Generator("mps").manual_seed(int(s))
        return pipe(prompt=PROMPTS[cls], num_inference_steps=steps,
                    guidance_scale=0.0, generator=g).images[0]

    return _run(make_image, "sdxl-turbo-1.0", out_dir, per_class, n_uninformative, seed)


def generate_gemini(out_dir: Path = IMG_DIR, per_class: int = 10, n_uninformative: int = 20,
                    seed: int = 0, model: str = "gemini-2.5-flash-image",
                    cost_per_image: float = 0.039) -> list[dict]:  # pragma: no cover
    """Real photoreal generation via Google Gemini image models.

    Requires a BILLED Gemini key — on the free tier every image model returns 429 limit:0
    (see docs/DECISIONS.md D22). Each image's cost is added to the shared $10 budget ledger.
    """
    import io
    import os

    from google import genai  # type: ignore
    from PIL import Image as PILImage

    from src.vlm.client import record_cost

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def make_image(cls: str, s: int):
        record_cost(cost_per_image)  # budget guard raises before spending if it would exceed $10
        resp = client.models.generate_content(model=model, contents=[PROMPTS[cls]])
        for part in resp.candidates[0].content.parts:
            data = getattr(getattr(part, "inline_data", None), "data", None)
            if data:
                return PILImage.open(io.BytesIO(data)).convert("RGB")
        raise RuntimeError(f"Gemini returned no image for class {cls!r}")

    return _run(make_image, model, out_dir, per_class, n_uninformative, seed)


def _write_manifest(manifest: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "file", "class", "kind", "prompt", "generator", "seed"])
        w.writeheader()
        w.writerows(manifest)


if __name__ == "__main__":
    import sys

    gen = sys.argv[1] if len(sys.argv) > 1 else "procedural"
    fn = {"procedural": generate_procedural, "sdxl": generate_sdxl, "gemini": generate_gemini}[gen]
    mf = fn()
    print(f"[{gen}] generated {len(mf)} images -> {IMG_DIR}; manifest -> {MANIFEST}")
