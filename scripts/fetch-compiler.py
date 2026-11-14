"""Fetch the self-contained compiler from Google's archived source repository.

No sudo or discontinued APT repository is required. Preserve upstream notices.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
import subprocess
import os

def download(url):
    if os.name == "nt":
        # Windows curl uses the system certificate store, including enterprise roots.
        return subprocess.check_output(["curl.exe", "--fail", "--location", "--silent", "--show-error", url])
    return urllib.request.urlopen(url, timeout=60).read()

COMMIT = "5020de9386ff370dcc1f63291a2d0f98eeb98adb"
ROOT = Path(__file__).resolve().parents[1]

def main():
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    base = "https://api.github.com/repos/google-coral/edgetpu/git/trees/" + COMMIT + "?recursive=1"
    tree = json.loads(download(base))
    files = [x for x in tree["tree"] if x["type"] == "blob" and x["path"].startswith("compiler/")]
    def fetch(item):
        path = ROOT / ".tools" / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/google-coral/edgetpu/" + COMMIT + "/" + item["path"]
        if not path.exists(): path.write_bytes(download(url))
        blob = path.read_bytes()
        git_sha = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        if git_sha != item["sha"]: raise ValueError("Upstream Git blob checksum mismatch: " + str(path))
        return {"path": item["path"], "sha256": hashlib.sha256(blob).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: records = list(pool.map(fetch, files))
    (ROOT / "reports" / "compiler-source.json").write_text(json.dumps(
        {"repository": "https://github.com/google-coral/edgetpu", "commit": COMMIT, "files": records}, indent=2), encoding="utf-8")
    compiler = ROOT / ".tools/compiler/x86_64/edgetpu_compiler"
    paths = [compiler, compiler.parent / "edgetpu_compiler_bin/edgetpu_compiler",
             compiler.parent / "edgetpu_compiler_bin/ld-linux-x86-64.so.2"]
    if os.name == "nt":
        for path in paths:
            linux = subprocess.check_output(["wsl", "-d", "Ubuntu", "--exec", "wslpath", "-a", path.as_posix()], text=True).strip()
            subprocess.run(["wsl", "-d", "Ubuntu", "--exec", "chmod", "+x", linux], check=True)
    else:
        for path in paths: path.chmod(path.stat().st_mode | 0o111)
    print("Compiler files verified against the pinned Google Git tree.")

if __name__ == "__main__": main()
