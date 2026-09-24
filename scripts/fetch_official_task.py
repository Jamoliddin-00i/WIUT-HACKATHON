from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

FILE_ID = "1lQlnjKNNBb5xWhGoyursq1GFajBIOkW7"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUT = Path(__file__).resolve().parents[1] / "docs" / "WIUT Hackathon _ CV Track Elimination Task.pdf"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    req = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        data = response.read()

    if not data.startswith(b"%PDF"):
        raise RuntimeError(
            "Google Drive did not return a PDF. Open the organizer Drive link in a browser, "
            "download the PDF manually, and save it at: " + str(OUT)
        )

    OUT.write_bytes(data)
    print(f"Saved {len(data):,} bytes to {OUT}")


if __name__ == "__main__":
    main()
