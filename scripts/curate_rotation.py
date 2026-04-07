import argparse
import json
import logging
import multiprocessing
import pathlib
import re
import sys
import tempfile
import time

import ollama
import pydantic

ROOT_DIR = pathlib.Path(__file__).parents[1]
RAW_DATA_DIR = ROOT_DIR / "raw_data"
TREATED_DATA_DIR = ROOT_DIR / "treated_data"

EXTRACTED_TEXT_DIR = RAW_DATA_DIR / "extracted"
INPUT_DIR = TREATED_DATA_DIR / "for_llm_curation" / "rotation"
ROTATION_RESULTS_DIR = TREATED_DATA_DIR / "rotation_results"
OUTPUT_DIR = TREATED_DATA_DIR / "curation_results" / "rotation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_PORT = 11434
LOG_INTERVAL = 60

PRE_CONTEXT_CHARS = 1400
POST_CONTEXT_CHARS_NO_NEXT = 600
MAX_EXCERPT_CHARS = 3200

UNSIGNED_REAL = r"\d+(?:\.\d+)?"
EXCESS_VALUE = re.compile(rf"({UNSIGNED_REAL})\s*%")
RATIO_VALUES = re.compile(rf"({UNSIGNED_REAL})\s*[:\/]\s*({UNSIGNED_REAL})")


SYSTEM_PROMPT = (
    "You are a chemist specializing in stereochemistry and asymmetric synthesis. "
    "Answer only based on the given excerpt, without making assumptions."
)

log = logging.getLogger("curate_rotation")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
)
_handler.flush = lambda: sys.stdout.flush()
log.addHandler(_handler)


class Response(pydantic.BaseModel):
    reasoning: str
    confidence_statement_1: int = pydantic.Field(ge=0, le=100)
    confidence_statement_2: int = pydantic.Field(ge=0, le=100)


class ExcessResponse(pydantic.BaseModel):
    reasoning: str
    there_is_evidence: bool
    excess_value: float = pydantic.Field(ge=0, le=100)


def parse_task_id(input_file):
    rel_parts = input_file.relative_to(INPUT_DIR).parts
    if len(rel_parts) != 3:
        raise ValueError(
            f"Expected <article>/<file_stem>/<start>.json under {INPUT_DIR}, got {input_file}"
        )
    article_id, file_stem, start_filename = rel_parts
    if not article_id.isdigit():
        raise ValueError(f"Article id must be numeric, got: {article_id}")
    try:
        block_start = int(pathlib.Path(start_filename).stem)
    except ValueError as e:
        raise ValueError(
            f"Start filename must be an integer stem, got: {start_filename}"
        ) from e
    return article_id, file_stem, block_start


def output_path(article_id, file_stem, block_start):
    return OUTPUT_DIR / article_id / file_stem / f"{block_start}.json"


def clean_text(text):
    return " ".join(text.replace("\f", " ").split())


def load_candidates(input_file):
    with open(input_file, "r") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError(f"Unsupported candidate JSON type: {type(raw)}")
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate entry at index {i} is not an object")
    return raw


def load_rotation_blocks(article_id, file_stem):
    folder = ROTATION_RESULTS_DIR / article_id / file_stem
    blocks = []
    for block_file in sorted(folder.glob("*.json"), key=lambda p: int(p.stem)):
        with open(block_file, "r") as f:
            block = json.load(f)
        if block.get("type") != "rotation":
            continue
        blocks.append(block)
    return blocks


def next_rotation_start(rotation_blocks, current_start):
    for block in rotation_blocks:
        if int(block["start"]) > current_start:
            return int(block["start"])
    return None


def build_excerpt(full_text, current_block, next_start):
    start = max(0, int(current_block["start"]) - PRE_CONTEXT_CHARS)
    if next_start is None:
        end = min(
            len(full_text), int(current_block["stop"]) + POST_CONTEXT_CHARS_NO_NEXT
        )
    else:
        end = min(len(full_text), next_start)
    if end <= start:
        end = min(len(full_text), start + MAX_EXCERPT_CHARS)
    if end - start > MAX_EXCERPT_CHARS:
        end = start + MAX_EXCERPT_CHARS
    return clean_text(full_text[start:end])


def parsed_excess_message(s: str) -> str:
    match = EXCESS_VALUE.search(s)
    if match is None:
        match = RATIO_VALUES.search(s)
        if match is None:
            return ""
        c1 = match.group(1)
        c2 = match.group(2)
        v1 = float(c1)
        v2 = float(c2)
        if (v1 + v2) == 0:
            return ""
        excess = round(100 * abs(v1 - v2) / (v1 + v2), 1)
        if excess == int(excess):
            excess = int(excess)
        cmax = c1 if v1 > v2 else c2
        cmin = c1 if v1 < v2 else c2
        return f"{excess}% [{c1}:{c2} → 100*({cmax}-{cmin})/({cmax}+{cmin})]"
    return f"{match.group(1).strip()}%"


def make_prompt(context, rotation_text, candidate):
    molecule = str(candidate.get("molecule", "unknown"))
    excess_or_ratio = str(candidate.get("excess_or_ratio", "unknown"))
    excess_message = parsed_excess_message(excess_or_ratio)
    context = context.replace(">", "")  # assume equality for all candidates
    return (
        f"{context}\n\n"
        "Read the excerpt above carefully and assign a confidence level from 0 to 100 "
        "to each statement.\n\n"
        f'1. The compound with optical rotation "{rotation_text}" is "{molecule}".\n\n'
        f'2. The stereoisomeric excess of "{molecule}" is precisely {excess_message}.\n'
    )


def call_llm(client, messages, model, pydantic_model):
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        format=pydantic_model.model_json_schema(),
        think=False,
        options={"temperature": 0},
    )
    parsed = pydantic_model.model_validate_json(response.message.content).model_dump()
    return parsed, response.message.content


def score_candidate(client, model, prompt, candidate):
    messages = [{"role": "user", "content": prompt}]
    try:
        response, assistant_content = call_llm(client, messages, model, Response)
        messages.append({"role": "assistant", "content": assistant_content})
        return (
            {
                **candidate,
                "reasoning": response["reasoning"],
                "major_isomer_confidence": response["confidence_statement_1"],
                "excess_confidence": response["confidence_statement_2"],
            },
            messages,
        )
    except Exception as e:
        return (
            {
                **candidate,
                "reasoning": f"Error during processing: {e}",
                "major_isomer_confidence": 0,
                "excess_confidence": 0,
            },
            messages,
        )


def resolve_excess(client, model, molecule, messages):
    prompt = (
        "1. Is there enough evidence in the text to extract the stereoisomeric "
        f'stereoisomeric excess of "{molecule}" with 100% confidence?\n'
        "2. If so, what is the actual value? If not, answer with 0.\n"
    )
    try:
        response, _ = call_llm(
            client,
            [*messages, {"role": "user", "content": prompt}],
            model,
            ExcessResponse,
        )
        return {
            "evident": bool(response["there_is_evidence"]),
            "value": float(response["excess_value"]),
            "reasoning": response["reasoning"],
        }
    except Exception as e:
        return {
            "evident": False,
            "value": 0,
            "reasoning": f"Error during processing: {e}",
        }


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2)
        pathlib.Path(tmp).rename(path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def curate_single_input(input_file, model, client, text_cache, rotations_cache):
    article_id, file_stem, block_start = parse_task_id(input_file)
    candidates = None
    try:
        candidates = load_candidates(input_file)
        if not candidates:
            raise ValueError("No candidates found")

        text_key = (article_id, file_stem)
        if text_key not in text_cache:
            text_path = EXTRACTED_TEXT_DIR / article_id / f"{file_stem}.txt"
            text_cache[text_key] = text_path.read_text()
        full_text = text_cache[text_key]

        if text_key not in rotations_cache:
            rotations_cache[text_key] = load_rotation_blocks(article_id, file_stem)
        rotation_blocks = rotations_cache[text_key]
        if not rotation_blocks:
            raise ValueError("No rotation blocks found in rotation_results")

        current_block = None
        for block in rotation_blocks:
            if int(block["start"]) == block_start:
                current_block = block
                break
        if current_block is None:
            raise ValueError(f"Rotation block start={block_start} not found")

        next_start = next_rotation_start(rotation_blocks, block_start)
        context = build_excerpt(full_text, current_block, next_start)
        rotation_text = str(current_block.get("text", "")).strip()

        curated = []
        for candidate in candidates:
            prompt = make_prompt(context, rotation_text, candidate)
            scored, messages = score_candidate(client, model, prompt, candidate)
            if (
                len(candidates) == 1
                and scored["major_isomer_confidence"] == 100
                and scored["excess_confidence"] < 100
            ):
                molecule = str(candidate.get("molecule", "unknown"))
                scored["excess_recovery"] = resolve_excess(
                    client, model, molecule, messages
                )
            curated.append(scored)
        return article_id, file_stem, block_start, curated, None
    except Exception as e:
        if candidates:
            fallback = [
                {
                    **candidate,
                    "reasoning": f"Error during processing: {e}",
                    "major_isomer_confidence": 0,
                    "excess_confidence": 0,
                }
                for candidate in candidates
            ]
            return article_id, file_stem, block_start, fallback, str(e)
        return article_id, file_stem, block_start, None, str(e)


def worker(tasks, model, port, error_queue):
    gpu_id = port - BASE_PORT
    total = len(tasks)
    client = ollama.Client(host=f"http://localhost:{port}", timeout=30)
    last_log = time.monotonic() - (LOG_INTERVAL + 1)

    text_cache = {}
    rotations_cache = {}

    for i, input_file in enumerate(tasks, 1):
        (
            article_id,
            file_stem,
            block_start,
            curated,
            error,
        ) = curate_single_input(input_file, model, client, text_cache, rotations_cache)
        if error:
            error_queue.put(f"{article_id}/{file_stem}/{block_start}: {error}")
        if curated is not None:
            write_json_atomic(output_path(article_id, file_stem, block_start), curated)

        now = time.monotonic()
        if now - last_log >= LOG_INTERVAL:
            log.info(f"GPU {gpu_id}: {i}/{total} ({100 * i / total:.1f}%)")
            last_log = now

    log.info(f"GPU {gpu_id}: done ({total}/{total})")


def discover_tasks():
    tasks = []
    ignored = 0
    for input_file in INPUT_DIR.glob("**/*.json"):
        try:
            article_id, file_stem, block_start = parse_task_id(input_file)
        except ValueError:
            ignored += 1
            continue
        if output_path(article_id, file_stem, block_start).exists():
            continue
        tasks.append(input_file)

    if ignored:
        log.info(
            f"Ignored {ignored} input file(s) that do not match "
            "expected <article>/<file_stem>/<start>.json layout"
        )
    tasks.sort()
    return tasks


def curate_rotation_data(num_servers=8, model="qwen3:14b"):
    tasks = discover_tasks()
    log.info(f"Found {len(tasks)} unprocessed curation task(s)")

    if not tasks:
        log.info("Nothing to do.")
        return

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
        log.warning(f"{len(errors)} task(s) failed:")
        for error in errors:
            log.warning(f"  {error}")
    else:
        log.info("All curation tasks processed successfully.")


def resolve_input_file(user_value):
    given = pathlib.Path(user_value)
    if given.exists():
        return given.resolve()
    candidate = (INPUT_DIR / given).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Input file not found: {user_value}")


def curate_single_file(input_file, model="qwen3:14b", port=BASE_PORT, to_stdout=False):
    client = ollama.Client(host=f"http://localhost:{port}", timeout=30)
    text_cache = {}
    rotations_cache = {}

    article_id, file_stem, block_start, curated, error = curate_single_input(
        input_file=input_file,
        model=model,
        client=client,
        text_cache=text_cache,
        rotations_cache=rotations_cache,
    )
    if curated is None:
        raise RuntimeError(f"Failed to process {input_file}: {error}")

    if to_stdout:
        payload = {
            "article_id": article_id,
            "file_stem": file_stem,
            "start": block_start,
            "results": curated,
        }
        print(json.dumps(payload, indent=2))
    else:
        out_file = output_path(article_id, file_stem, block_start)
        write_json_atomic(out_file, curated)
        log.info(f"Wrote curation output: {out_file}")

    if error:
        log.warning(f"{article_id}/{file_stem}/{block_start}: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate optical rotation molecule/excess combinations with LLM"
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
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help=(
            "Process a single input JSON file "
            "(absolute path or path relative to treated_data/for_llm_curation/rotation)"
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print single-file result JSON to stdout instead of writing to disk",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=BASE_PORT,
        help=f"Ollama port for single-file mode (default: {BASE_PORT})",
    )
    args = parser.parse_args()
    if args.stdout and not args.input_file:
        parser.error("--stdout requires --input-file")

    if args.input_file:
        input_file = resolve_input_file(args.input_file)
        curate_single_file(
            input_file=input_file,
            model=args.model,
            port=args.port,
            to_stdout=args.stdout,
        )
    else:
        curate_rotation_data(num_servers=args.num_servers, model=args.model)
