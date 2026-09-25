"""Download pinned, unsplit ShareGPT conversations and retain source provenance."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from spark_run import ROOT, save

REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
FILES = [f"HTML_cleaned_raw_dataset/sg_90k_part{i}_html_cleaned.json" for i in (1, 2)]


def prepare(output):
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Dataset output must be outside the source repository")
    output.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    for name in FILES:
        path = output / Path(name).name
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"
        if not path.exists():
            temporary = path.with_suffix(".download")
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            temporary.replace(path)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        part = json.loads(path.read_text())
        records.extend(part)
        sources.append(dict(file=name, url=url, sha256=checksum, records=len(part)))
        print(f"Loaded {name}: {len(part)} records", flush=True)
    target = output / "sharegpt_unsplit.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False))
    temporary.replace(target)
    with target.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    save(
        target.with_suffix(".metadata.json"),
        dict(
            repo=REPO,
            revision=REVISION,
            sources=sources,
            sha256=checksum,
            records=len(records),
            transformation="Concatenate upstream HTML-cleaned parts; no splitting, truncation, or length filtering",
        ),
    )
    print(target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
