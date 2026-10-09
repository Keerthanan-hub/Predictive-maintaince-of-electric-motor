"""Populate a backend data directory with supported 12 kHz reference recordings."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

RECORDING_IDS = (
    97, 98, 99, 100,
    *range(105, 109), *range(169, 173), *range(209, 213),
    *range(118, 122), *range(185, 189), *range(222, 226),
    *range(130, 134), *range(197, 201), *range(234, 238),
)
DOWNLOAD_URL = "https://engineering.case.edu/sites/default/files/{recording_id}.mat"


def download(recording_id, destination):
    path = destination / f"{recording_id}.mat"
    if path.is_file() and path.stat().st_size > 100_000:
        return f"{path.name}: already present"
    request = Request(DOWNLOAD_URL.format(recording_id=recording_id),
                      headers={"User-Agent": "Bearing Monitor research prototype"})
    with urlopen(request, timeout=90) as response:
        content = response.read()
    if len(content) < 100_000 or not content.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError(f"Unexpected content while downloading {path.name}")
    temporary = path.with_suffix(".mat.part")
    temporary.write_bytes(content)
    temporary.replace(path)
    return f"{path.name}: downloaded ({len(content)} bytes)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data") / "cwru")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download, recording_id, args.output)
                   for recording_id in RECORDING_IDS]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    print(f"Validated {len(RECORDING_IDS)} reference recordings in {args.output}")


if __name__ == "__main__":
    main()
