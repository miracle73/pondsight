"""Download a 30s public underwater fish clip for testing.

Falls back to a general-purpose sample clip if the fish clip URL is unavailable.
Place your own .mp4 at data/sample.mp4 to use your own footage.
"""
import urllib.request
from pathlib import Path

FISH_URL = "https://cdn.pixabay.com/video/2023/03/26/155542-810170145_large.mp4"
FALLBACK_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/1080/Big_Buck_Bunny_1080_10s_1MB.mp4"


def _fetch(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) < 100_000:
            return False
        dest.write_bytes(data)
        print(f"Saved clip to {dest} ({len(data)//1024} KB)")
        return True
    except Exception as e:
        print(f"Download failed for {url}: {e}")
        return False


def download():
    dest = Path("data/sample.mp4")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"{dest} already exists.")
        return
    if not _fetch(FISH_URL, dest):
        print("Fish clip unavailable, using fallback sample clip...")
        _fetch(FALLBACK_URL, dest)


if __name__ == "__main__":
    download()
