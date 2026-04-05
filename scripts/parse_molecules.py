import argparse
import json
import logging
import multiprocessing
import pathlib
import sys
import tempfile
import time

import ollama
import pydantic

ROOT_DIR = pathlib.Path(__file__).parents[1]
RAW_DATA_DIR = ROOT_DIR / "raw_data"
TREATED_DATA_DIR = ROOT_DIR / "treated_data"

EXTRACTED_TEXT_DIR = RAW_DATA_DIR / "extracted"
DATA_BLOCKS_DIR = TREATED_DATA_DIR / "data_blocks"

PROMPT_LENGTH = 2096
BASE_PORT = 11434
LOG_INTERVAL = 60

RESULTS_DIR = TREATED_DATA_DIR / "rotation_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("parse_molecules")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
)
_handler.flush = lambda: sys.stdout.flush()
log.addHandler(_handler)


class MoleculeData(pydantic.BaseModel):
    iupac_name: str
    excess_or_ratio: str


def make_prompt(full_text, block):
    stop = block["stop"]

    prompt = f"""\
{full_text[:stop]}.

Copy the exact IUPAC name (including stereochemistry) of the molecule with the final \
optical rotation above.
If not found or ambiguous, refrain from guessing and return "unknown".
Also return the enantiomeric/diastereomeric excess or ratio, if available, or \
"unknown" if not available.
"""

    start = max(0, len(prompt) - PROMPT_LENGTH)
    return prompt[start:]


def result_path(article_id, file_stem, block_start):
    return RESULTS_DIR / article_id / file_stem / f"{block_start}.json"


def call_llm(client, prompt, model):
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=MoleculeData.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    return MoleculeData.model_validate_json(response.message.content).model_dump()


def worker(tasks, model, port, error_queue):
    gpu_id = port - BASE_PORT
    total = len(tasks)
    client = ollama.Client(host=f"http://localhost:{port}", timeout=20)
    last_log = time.monotonic() - (LOG_INTERVAL + 1)
    cached_file = None
    cached_text = None
    for i, (block, text_file) in enumerate(tasks, 1):
        article_id = text_file.parent.name
        file_stem = text_file.stem
        try:
            if text_file != cached_file:
                cached_file = text_file
                cached_text = text_file.read_text()
            prompt = make_prompt(cached_text, block)
            result = call_llm(client, prompt, model)
            output = {
                **block,
                "molecule": result["iupac_name"],
                "excess_or_ratio": result["excess_or_ratio"],
            }
            out_file = result_path(article_id, file_stem, block["start"])
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=out_file.parent, suffix=".tmp")
            try:
                with open(fd, "w") as f:
                    json.dump(output, f, indent=2)
                pathlib.Path(tmp).rename(out_file)
            except BaseException:
                pathlib.Path(tmp).unlink(missing_ok=True)
                raise
        except Exception as e:
            error_queue.put(f"{article_id}/{file_stem}/block@{block['start']}: {e}")

        now = time.monotonic()
        if now - last_log >= LOG_INTERVAL:
            log.info(f"GPU {gpu_id}: {i}/{total} ({100 * i / total:.1f}%)")
            last_log = now

    log.info(f"GPU {gpu_id}: done ({total}/{total})")


def parse_molecules(num_servers=8, model="qwen3:14b"):
    data_block_files = list(DATA_BLOCKS_DIR.glob("**/*.json"))
    log.info(f"Found {len(data_block_files)} data block files")

    tasks = []
    for dbf in data_block_files:
        article_id = dbf.parent.name
        text_file = EXTRACTED_TEXT_DIR / article_id / f"{dbf.stem}.txt"
        with open(dbf, "r") as f:
            data_blocks = json.load(f)
        for block in data_blocks:
            if block["type"] != "rotation":
                continue
            if result_path(article_id, text_file.stem, block["start"]).exists():
                continue
            tasks.append((block, text_file))

    log.info(f"Found {len(tasks)} unprocessed rotation blocks")

    if not tasks:
        log.info("Nothing to do.")
        return

    tasks.sort(key=lambda t: t[1])
    sublists = [tasks[i::num_servers] for i in range(num_servers)]

    error_queue = multiprocessing.Queue()
    processes = []
    for i in range(num_servers):
        if not sublists[i]:
            continue
        p = multiprocessing.Process(
            target=worker,
            args=(sublists[i], model, BASE_PORT + i, error_queue),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get_nowait())

    if errors:
        log.warning(f"{len(errors)} block(s) failed:")
        for error in errors:
            log.warning(f"  {error}")
    else:
        log.info("All blocks processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse molecules from data blocks using LLM"
    )
    parser.add_argument(
        "-n",
        "--num-servers",
        type=int,
        default=8,
        help="Number of Ollama servers/GPUs to use (default: 8)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen3:14b",
        help="Ollama model to use (default: qwen3:14b)",
    )
    args = parser.parse_args()
    parse_molecules(num_servers=args.num_servers, model=args.model)
