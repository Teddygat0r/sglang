"""Measure TTFT with and without mamba prefix cache hits.

Sends a discarded warmup request, then a cold test request, then the same test
request several more times so the mamba prefix cache serves it. Prints TTFT
and end-to-end latency per run.

Usage:
    python scripts/playground/measure_mamba_cache_ttft.py
    python scripts/playground/measure_mamba_cache_ttft.py --url http://localhost:30000 --warm-runs 3
"""

import argparse
import time

import requests

WARMUP = (
    "In the early nineteenth century, a young naturalist named Charles Darwin set sail aboard "
    "the HMS Beagle on a voyage that would fundamentally change our understanding of life on "
    "Earth. Over five years, Darwin collected specimens, observed geological formations, and "
    "documented the diverse flora and fauna of South America, the Galapagos Islands, Australia, "
    "and many other locations. His meticulous notes and collected samples would eventually form "
    "the empirical foundation for his theory of evolution by natural selection, which he would "
    "not publish until more than two decades after returning home. The most famous of his "
    "observations concerned the finches of the Galapagos Islands, whose beaks varied subtly "
    "from island to island in ways that suggested adaptation to local food sources. These "
    "observations, combined with his reading of Malthus on population growth and his study of "
    "animal breeding, led Darwin to propose that species change over time through a gradual "
    "process of differential survival and reproduction. The publication of On the Origin of "
    "Species in 1859 sparked immediate controversy, both within the scientific community and "
    "among religious authorities, but within a generation the basic framework of evolutionary "
    "theory was accepted by nearly all biologists. The most controversial figure in this debate was"
)

TEST = (
    "Photosynthesis is the biochemical process by which plants, algae, and certain bacteria "
    "convert sunlight, water, and carbon dioxide into glucose and oxygen, forming the foundation "
    "of nearly all food chains on Earth. The process takes place primarily in specialized "
    "organelles called chloroplasts, which are found in the green tissues of plants and contain "
    "the pigment chlorophyll that gives leaves their characteristic color. Chlorophyll absorbs "
    "light most efficiently in the red and blue portions of the visible spectrum, reflecting green "
    "light, which is why most plants appear green to the human eye. Photosynthesis consists of two "
    "interconnected sets of reactions: the light-dependent reactions, which occur in the thylakoid "
    "membranes of the chloroplast and produce ATP and NADPH while releasing oxygen as a byproduct, "
    "and the Calvin cycle, which takes place in the stroma and uses ATP and NADPH to fix carbon "
    "dioxide into three-carbon sugars that are later assembled into glucose and other carbohydrates. "
    "The overall chemical equation is six molecules of water and six molecules of carbon dioxide "
    "combining, in the presence of light energy, to produce one molecule of glucose and six "
    "molecules of oxygen. The evolution of oxygenic photosynthesis roughly two and a half billion "
    "years ago transformed Earth's atmosphere from an anaerobic environment into one rich in "
    "oxygen, an event known as the Great Oxidation Event, which paved the way for the evolution "
    "of complex multicellular life. Modern research on photosynthesis has focused on engineering "
    "more efficient crop plants, developing artificial photosynthetic systems to produce clean "
    "fuels, and understanding how plants respond to climate stress. The single biggest obstacle "
    "preventing artificial photosynthesis from becoming economically viable today is"
) * 10


def run(label: str, url: str, prompt: str, max_tokens: int) -> None:
    t0 = time.time()
    r = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
        },
        stream=True,
    )
    ttft = None
    last = None
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        if b"[DONE]" in line:
            break
        if ttft is None:
            ttft = time.time() - t0
        last = time.time()
    total = (last - t0) if last else (time.time() - t0)
    ttft_ms = ttft * 1000 if ttft is not None else float("nan")
    print(f"{label:<20s}  TTFT={ttft_ms:7.1f} ms   E2E={total * 1000:7.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:30000")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warm-runs", type=int, default=3)
    args = parser.parse_args()

    run("warmup (discard)", args.url, WARMUP, args.max_tokens)
    run("run 1 (cold)", args.url, TEST, args.max_tokens)
    for i in range(args.warm_runs):
        run(f"run {i + 2} (warm)", args.url, TEST, args.max_tokens)


if __name__ == "__main__":
    main()
