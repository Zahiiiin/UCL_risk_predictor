"""
Builds one self-contained HTML file (page, styles, scripts, fonts and both trained
models all inside) so the site can be opened as a single page with no server.

Run from the project folder:   python tools/build_preview.py
Writes preview/ucl_risk_predictor_preview.html
"""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "static"


def font_uri(name):
    return "data:font/woff2;base64," + base64.b64encode((S / "fonts" / name).read_bytes()).decode("ascii")


def main():
    css = (S / "style.css").read_text()
    css = re.sub(r'url\("fonts/([^"]+)"\)', lambda m: f'url("{font_uri(m.group(1))}")', css)

    html = (S / "index.html").read_text()
    body = html[html.index("<body>") + len("<body>"):html.index("</body>")]
    body = re.sub(r'\s*<script src="[^"]+"></script>', "", body)
    models = (S / "browser_models.json").read_text()
    engine = (S / "local-engine.js").read_text()
    app = (S / "app.js").read_text()
    for name, text in (("models", models), ("engine", engine), ("app", app)):
        assert "</script" not in text, name

    page = (
        "<title>UCL Risk Predictor</title>\n"
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        f"<style>\n{css}\n</style>\n"
        f"{body}\n"
        f'<script type="application/json" id="ucl-models">{models}</script>\n'
        f"<script>\n{engine}\n</script>\n"
        f"<script>\n{app}\n</script>\n"
    )
    out = ROOT / "preview"
    out.mkdir(exist_ok=True)
    path = out / "ucl_risk_predictor_preview.html"
    path.write_text(page)
    print(f"Wrote {path.relative_to(ROOT)} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
