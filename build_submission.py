"""Render the ONE ordered BITS Assignment 2 PDF after genuine lab work.

Final output intentionally requires real GitHub / Streamlit URLs and a BITS
Virtual Lab screenshot. A separately-labelled draft can be created earlier.
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
from urllib.parse import urlparse

import mistune
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent


def valid_url(value: str) -> bool:
    try:
        p = urlparse(value)
        return p.scheme == "https" and bool(p.netloc) and "." in p.netloc
    except ValueError:
        return False


def screenshot_html(path: Path | None, draft: bool) -> str:
    if path is None:
        return '<div class="missing">PENDING: insert ONE genuine screenshot from BITS Virtual Lab.</div>'
    if not path.exists():
        raise FileNotFoundError(f"Screenshot not found: {path}")
    if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        raise ValueError("Screenshot must be a PNG or JPG file.")
    content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="proof" src="data:{content_type};base64,{encoded}" alt="BITS lab execution proof" />'


def create_pdf(repo_url: str, app_url: str, shot: Path | None, output: Path, draft=False):
    if not draft:
        if not repo_url.startswith("https://github.com/"):
            raise ValueError("A real GitHub repository URL is required for the final PDF.")
        if not valid_url(app_url) or "streamlit.app" not in urlparse(app_url).netloc:
            raise ValueError("A real Streamlit Community Cloud URL is required for the final PDF.")
        if shot is None:
            raise ValueError("One real BITS Virtual Lab screenshot is required for the final PDF.")
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError("README.md missing. Run build_readme.py after training.")
    readme = readme_path.read_text(encoding="utf-8")
    if not draft:
        if any(marker in readme for marker in ["[ADD GITHUB", "[ADD LIVE STREAMLIT", "[NOT TRAINED", "PENDING", "Not yet measured", "WORKING REFERENCE", "[ADD METRIC"]):
            raise ValueError("README contains unfilled placeholders. Finish README before final PDF.")
        if repo_url not in readme or app_url not in readme:
            raise ValueError("README links must match the final submission links; regenerate README.")
    markdown = mistune.create_markdown(plugins=["table", "strikethrough"])
    html_readme = markdown(readme)
    shot_html = screenshot_html(shot, draft)
    repo = html.escape(repo_url)
    app = html.escape(app_url)
    note = ('<p class="status">DRAFT ONLY - NOT FOR SUBMISSION. Links, lab proof or real model '
            'metrics may be incomplete.</p>' if draft else '')
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<style>
@page {{size: A4; margin: 17mm 17mm 18mm 17mm; @bottom-right {{ content: 'Page ' counter(page); color:#64748b; font-size:9px;}}}}
body {{font-family:'DejaVu Sans',sans-serif; color:#243247; font-size:10px; line-height:1.46;}}
h1 {{font-size:19px; color:#123447; margin:0 0 12px;}}
h2 {{font-size:13.2px;color:#0D6267;margin:19px 0 8px;break-after:avoid;}}
h3 {{font-size:11.2px;color:#2C4056;margin:15px 0 6px;break-after:avoid;}}
p {{margin:7px 0 10px;}}
li {{margin:4px 0;}}
a {{color:#096E70; text-decoration:underline;overflow-wrap:anywhere;}}
code {{background:#F0F4F6; font-family:'DejaVu Sans Mono',monospace;font-size:8.4px;overflow-wrap:anywhere;}}
pre {{white-space:pre-wrap;overflow-wrap:anywhere;}}
table {{width:100%;border-collapse:collapse; font-size:8.2px; margin:11px 0 15px; break-inside:auto;}}
th {{background:#E4EEEB; color:#123447; font-weight:bold;}}
th,td {{padding:6px 6px;border:0.7px solid #D9E2E7; vertical-align:top;overflow-wrap:anywhere;}}
tr {{break-inside:avoid;}}
article.readme {{break-before:page;}}
.links {{border:1px solid #B9CECF; border-radius:6px; padding:12px 14px; margin:12px 0;}}
.eyebrow {{color:#0D7779;font-size:9px;letter-spacing:1.5px;font-weight:700;margin-top:12px;}}
.proof {{max-height:170mm;max-width:100%;height:auto;object-fit:contain;display:block;border:1px solid #D1DBDF;}}
.proof-section {{break-before:page;}}
.status, .missing {{padding:14px;background:#FFF0DE;color:#78420C;border:1px solid #E6B978;border-radius:5px;}}
.foot {{color:#556273;font-size:9px;}}
</style></head><body>
<div class="eyebrow">BITS PILANI WILP / MACHINE LEARNING ASSIGNMENT 2</div>
<h1>Steel Plate Surface Fault Classification</h1>
{note}
<p class="foot">Ordered submission contents, following the assignment handout: 1. GitHub link, 2. live Streamlit link, 3. BITS Lab screenshot, 4. complete GitHub README content.</p>
<section class="links">
<h2>1. GitHub Repository Link</h2><p><a href="{repo}">{repo}</a></p>
<p>Complete source code, requirements, test CSV, README and five fitted models are stored in this repository.</p>
<h2>2. Live Streamlit App Link</h2><p><a href="{app}">{app}</a></p>
<p>Opens the interactive inference frontend on Streamlit Community Cloud.</p>
</section>
<section class="proof-section"><h2>3. Proof of execution on BITS Virtual Lab</h2>
{shot_html}
<p class="foot">Exactly one actual screenshot required. Do not substitute a local computer or fabricated screenshot.</p></section>
<article class="readme"><h2>4. GitHub README.md - Complete Content</h2>
{html_readme}</article>
</body></html>"""
    HTML(string=document, base_url=str(ROOT)).write_pdf(str(output))
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github-url", default="[ADD GITHUB REPOSITORY URL]")
    ap.add_argument("--app-url", default="[ADD LIVE STREAMLIT URL]")
    ap.add_argument("--screenshot", type=Path, default=None)
    ap.add_argument("--draft", action="store_true", help="Only create a labelled incomplete preview")
    ap.add_argument("--output", type=Path, default=None)
    o = ap.parse_args()
    target = o.output or ROOT / ("submission_DRAFT.pdf" if o.draft else "submission_FINAL.pdf")
    create_pdf(o.github_url, o.app_url, o.screenshot, target, o.draft)


if __name__ == "__main__":
    main()
