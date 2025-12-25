import argparse
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"].rstrip("/")
AZURE_KEY = os.environ["AZURE_VISION_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------- Azure Read OCR (preferred: Image Analysis 4.0) ----------
def azure_read_imageanalysis(image_bytes: bytes) -> str:
    """
    Uses Azure AI Vision 'Image Analysis' endpoint with Read feature.
    Works with Computer Vision resources that support:
      POST {endpoint}/computervision/imageanalysis:analyze?api-version=2023-02-01-preview&features=read
    """
    url = f"{AZURE_ENDPOINT}/computervision/imageanalysis:analyze"
    params = {
        "api-version": "2023-02-01-preview",
        "features": "read",
    }
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream",
    }

    r = requests.post(url, params=params, headers=headers, data=image_bytes, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Expected shape includes: data["readResult"]["content"] (full text)
    read_result = data.get("readResult") or {}
    content = read_result.get("content")
    if content:
        return content.strip()

    # Fallback parse if only lines exist
    lines = []
    for block in read_result.get("blocks", []) or []:
        for line in block.get("lines", []) or []:
            text = line.get("text")
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


# ---------- Azure Read OCR (fallback: Read 3.2 async) ----------
def azure_read_v32(image_bytes: bytes) -> str:
    """
    Fallback to older Read API:
      POST {endpoint}/vision/v3.2/read/analyze
      Poll Operation-Location until succeeded.
    """
    submit_url = f"{AZURE_ENDPOINT}/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/octet-stream",
    }

    submit = requests.post(submit_url, headers=headers, data=image_bytes, timeout=60)
    submit.raise_for_status()
    op_loc = submit.headers.get("Operation-Location")
    if not op_loc:
        raise RuntimeError("Azure Read v3.2: missing Operation-Location header.")

    # Poll
    for _ in range(60):
        time.sleep(0.5)
        poll = requests.get(op_loc, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY}, timeout=60)
        poll.raise_for_status()
        result = poll.json()
        status = (result.get("status") or "").lower()
        if status == "succeeded":
            lines = []
            analyze = result.get("analyzeResult", {})
            for page in analyze.get("readResults", []) or []:
                for line in page.get("lines", []) or []:
                    txt = line.get("text")
                    if txt:
                        lines.append(txt)
            return "\n".join(lines).strip()
        if status == "failed":
            raise RuntimeError(f"Azure Read v3.2 failed: {result}")
    raise TimeoutError("Azure Read v3.2 timed out while polling.")


def azure_handwriting_ocr(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()

    # Try Image Analysis 4.0 first; fallback to v3.2 if your resource/region doesn’t support it.
    try:
        text = azure_read_imageanalysis(image_bytes)
        if text:
            return text
    except requests.HTTPError:
        pass

    return azure_read_v32(image_bytes)


# ---------- OpenAI cleanup ----------
def openai_cleanup(raw_text: str) -> str:
    if not OPENAI_API_KEY:
        return raw_text

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = (
        "You are cleaning OCR output from handwritten dance performance observation notebooks.\n"
        "Rules:\n"
        "- Preserve meaning; do not invent details.\n"
        "- Fix obvious OCR artifacts (broken words, random line breaks).\n"
        "- Keep original ordering.\n"
        "- Output as Markdown.\n"
        "- If a date is present, put it as a top-level heading.\n"
        "- Use bullets where appropriate.\n\n"
        "OCR TEXT:\n"
        f"{raw_text}"
    )

    # Use a current general-purpose text model available to your account.
    # If you know the exact model name you want, replace the string below.
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return resp.output_text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str, help="Path to an image (jpg/png) of a notebook page")
    ap.add_argument("--outdir", type=str, default="out", help="Output directory")
    ap.add_argument("--no-clean", action="store_true", help="Skip OpenAI cleanup")
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw_text = azure_handwriting_ocr(image_path)

    raw_path = outdir / f"{image_path.stem}.raw.txt"
    raw_path.write_text(raw_text, encoding="utf-8")

    if args.no_clean:
        print(f"Wrote: {raw_path}")
        return

    cleaned = openai_cleanup(raw_text)
    clean_path = outdir / f"{image_path.stem}.clean.md"
    clean_path.write_text(cleaned, encoding="utf-8")

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {clean_path}")


if __name__ == "__main__":
    main()
