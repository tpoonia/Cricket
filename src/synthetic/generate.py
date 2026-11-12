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

# Strong per-class prompts for the real SDXL path (kept in the manifest).
PROMPTS = {
    "green": "close-up photograph of a fresh green grassy cricket pitch, moist surface, overcast morning, 35mm",
    "dry": "close-up photograph of a dry cracked brown cricket pitch, visible cracks, sunbaked, harsh midday light",
    "flat": "close-up photograph of a hard flat even light-brown cricket pitch, well-rolled, clear day",
    "slow": "close-up photograph of a dusty worn slow cricket pitch, grey-brown dust, scuff marks, subcontinental ground",
    "wet": "photograph of a cricket pitch under covers, wet waterlogged outfield, puddles, rain",
    "clear": "photograph of a clear blue sky over a cricket stadium, bright sunshine",
    "overcast": "photograph of a heavy grey overcast sky over a cricket ground, thick clouds",
    "rain_now": "photograph of rain falling over a cricket ground, wet dark sky, drizzle",
    "night": "photograph of a floodlit cricket stadium at night, bright floodlights, dark sky",
    "uninformative": "wide blurry photograph of a packed cricket stadium crowd, out of focus",
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


def generate_procedural(out_dir: Path = IMG_DIR, per_class: int = 10,
                        n_uninformative: int = 20, seed: int = 0) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    sid = 0
    plan = [(c, "pitch") for c in PITCH_CLASSES] + [(c, "sky") for c in SKY_CLASSES]
    for cls, kind in plan:
        for i in range(per_class):
            sid += 1
            s = seed * 1000 + sid
            img = _render(cls, s)
            name = f"{kind}_{cls}_{i:02d}.png"
            img.save(out_dir / name)
            manifest.append({"id": name[:-4], "file": name, "class": cls, "kind": kind,
                             "prompt": PROMPTS[cls], "generator": "procedural-v1", "seed": s})
    for i in range(n_uninformative):
        sid += 1
        s = seed * 1000 + sid
        img = _render(UNINFORMATIVE, s)
        name = f"uninformative_{i:02d}.png"
        img.save(out_dir / name)
        manifest.append({"id": name[:-4], "file": name, "class": UNINFORMATIVE, "kind": "other",
                         "prompt": PROMPTS[UNINFORMATIVE], "generator": "procedural-v1", "seed": s})
    _write_manifest(manifest)
    return manifest


def _write_manifest(manifest: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "file", "class", "kind", "prompt", "generator", "seed"])
        w.writeheader()
        w.writerows(manifest)


def generate_sdxl(out_dir: Path = IMG_DIR, per_class: int = 10, seed: int = 0):  # pragma: no cover
    """Real photoreal generation (gated). Requires torch+diffusers on MPS."""
    from diffusers import StableDiffusionXLPipeline  # type: ignore
    import torch  # type: ignore

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    ).to("mps")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cls, prompt in PROMPTS.items():
        for i in range(per_class):
            g = torch.Generator("mps").manual_seed(seed + i)
            img = pipe(prompt, num_inference_steps=30, generator=g).images[0]
            name = f"{cls}_{i:02d}.png"
            img.save(out_dir / name)
            manifest.append({"id": name[:-4], "file": name, "class": cls, "kind": "img",
                             "prompt": prompt, "generator": "sdxl-base-1.0", "seed": seed + i})
    _write_manifest(manifest)
    return manifest


if __name__ == "__main__":
    mf = generate_procedural()
    print(f"generated {len(mf)} images -> {IMG_DIR}; manifest -> {MANIFEST}")
