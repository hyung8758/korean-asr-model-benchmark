"""Prepare a simple Whisper benchmark manifest from Korean STT corpora."""

import argparse
import json
import logging
import random
import re
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torchaudio
from tqdm import tqdm

from stt_benchmark.text import normalize_text

LOGGER = logging.getLogger("prepare_whisper_benchmark_data")
AUDIO_SUFFIXES = {".wav", ".flac"}
DATASETS = ("zeroth", "pansori_tedxkr", "asr_kcsc")
BUCKETS = ("short", "mid", "long")


@dataclass(frozen=True)
class Candidate:
    utt_id: str
    dataset: str
    audio_path: Path
    text_raw: str
    speaker: str
    split: str
    source_text: Path
    start_sec: float | None = None
    end_sec: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Whisper benchmark wavs, manifest.jsonl, and summary.json."
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_hours_per_corpus", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write_kaldi", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def write_jsonl_line(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_drop(handle, counter: Counter, base: dict, reason: str, **extra: object) -> None:
    counter[reason] += 1
    write_jsonl_line(handle, {**base, "reason": reason, **extra})


def bucket_for_duration(duration: float) -> str:
    if duration < 15.0:
        return "short"
    if duration < 300.0:
        return "mid"
    return "long"


def safe_id(dataset: str, utt_id: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", utt_id).strip("_")
    return f"{dataset}_{cleaned}"


def infer_split(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for split in ("train", "dev", "test"):
        if any(split in part for part in parts):
            return split
    return "unknown"


def infer_corpus_key(path: Path) -> str | None:
    name = path.name.lower()
    if "zeroth" in name or "zeroth_korean" in name:
        return "zeroth"
    if "pansori" in name or "tedxkr" in name or "tedx" in name:
        return "pansori_tedxkr"
    if "kcsc" in name or "conversational" in name or "asr-kcsc" in name:
        return "asr_kcsc"
    return None


def extract_corpus_archives(data_root: Path) -> None:
    """Extract known corpus archives before directory discovery."""
    extract_root = data_root / "_extracted_corpora"
    for archive_path in sorted(data_root.iterdir()):
        if archive_path.is_dir():
            continue
        lower_name = archive_path.name.lower()
        if infer_corpus_key(archive_path) is None:
            continue
        if not (
            lower_name.endswith(".zip")
            or lower_name.endswith(".tar.gz")
            or lower_name.endswith(".tgz")
            or lower_name.endswith(".tar")
        ):
            continue

        target_dir = extract_root / archive_path.stem.replace(".tar", "")
        marker = target_dir / ".extract_complete"
        if marker.exists():
            continue

        LOGGER.info("Extracting %s to %s", archive_path, target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(target_dir)
        else:
            with tarfile.open(archive_path) as tf:
                tf.extractall(target_dir)
        marker.write_text("ok\n", encoding="utf-8")


def discover_corpus_dirs(data_root: Path) -> dict[str, list[Path]]:
    extract_corpus_archives(data_root)
    discovered: dict[str, list[Path]] = {dataset: [] for dataset in DATASETS}
    for path in sorted(data_root.rglob("*")):
        if not path.is_dir():
            continue
        key = infer_corpus_key(path)
        if key is not None:
            discovered[key].append(path)
    return discovered


def read_transcript_lines(path: Path) -> Iterable[tuple[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]


def parse_trans_txt_corpus(corpus_dir: Path, dataset: str, speaker_level: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for trans_path in sorted(corpus_dir.rglob("*.trans.txt")):
        speaker = trans_path.parents[speaker_level].name
        audio_by_stem = {
            audio_path.stem: audio_path
            for audio_path in trans_path.parent.iterdir()
            if audio_path.suffix.lower() in AUDIO_SUFFIXES
        }
        for utt_id, text in read_transcript_lines(trans_path):
            audio_path = audio_by_stem.get(utt_id)
            if audio_path is None:
                continue
            candidates.append(
                Candidate(
                    utt_id=utt_id,
                    dataset=dataset,
                    audio_path=audio_path,
                    text_raw=text,
                    speaker=speaker,
                    split=infer_split(trans_path),
                    source_text=trans_path,
                )
            )
    return candidates


def parse_zeroth(corpus_dir: Path) -> list[Candidate]:
    return parse_trans_txt_corpus(corpus_dir, "zeroth", speaker_level=0)


def parse_pansori_tedxkr(corpus_dir: Path) -> list[Candidate]:
    return parse_trans_txt_corpus(corpus_dir, "pansori_tedxkr", speaker_level=1)


def parse_kcsc_line(line: str) -> tuple[float, float, str, str, str] | None:
    match = re.match(r"^\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+(\S+)\s+(.*)$", line.strip())
    if not match:
        return None
    start_sec = float(match.group(1))
    end_sec = float(match.group(2))
    speaker = match.group(3)
    gender = match.group(4)
    text = match.group(5).strip()
    if speaker == "0" and gender == "none":
        speaker = "unknown"
    return start_sec, end_sec, speaker, gender, text


def parse_asr_kcsc(corpus_dir: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    txt_dirs = [path for path in corpus_dir.rglob("TXT") if path.is_dir()]
    wav_dirs = [path for path in corpus_dir.rglob("WAV") if path.is_dir()]
    if not txt_dirs or not wav_dirs:
        return candidates

    wav_by_stem: dict[str, Path] = {}
    for wav_dir in wav_dirs:
        for wav_path in wav_dir.rglob("*.wav"):
            wav_by_stem.setdefault(wav_path.stem, wav_path)

    for txt_dir in txt_dirs:
        for txt_path in sorted(txt_dir.glob("*.txt")):
            audio_path = wav_by_stem.get(txt_path.stem)
            if audio_path is None:
                continue
            with txt_path.open("r", encoding="utf-8", errors="replace") as handle:
                for idx, line in enumerate(handle):
                    parsed = parse_kcsc_line(line)
                    if parsed is None:
                        continue
                    start_sec, end_sec, speaker, _gender, text = parsed
                    candidates.append(
                        Candidate(
                            utt_id=f"{txt_path.stem}_{idx:05d}_{int(start_sec * 1000):08d}_{int(end_sec * 1000):08d}",
                            dataset="asr_kcsc",
                            audio_path=audio_path,
                            text_raw=text,
                            speaker=speaker,
                            split=infer_split(txt_path),
                            source_text=txt_path,
                            start_sec=start_sec,
                            end_sec=end_sec,
                        )
                    )
    return candidates


def parse_candidates(corpus_dirs: dict[str, list[Path]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    parsers = {
        "zeroth": parse_zeroth,
        "pansori_tedxkr": parse_pansori_tedxkr,
        "asr_kcsc": parse_asr_kcsc,
    }
    for dataset, dirs in corpus_dirs.items():
        seen_audio_roots: set[Path] = set()
        for corpus_dir in dirs:
            resolved = corpus_dir.resolve()
            if any(parent in seen_audio_roots for parent in resolved.parents):
                continue
            dataset_candidates = parsers[dataset](corpus_dir)
            if dataset_candidates:
                LOGGER.info("Parsed %d candidates from %s", len(dataset_candidates), corpus_dir)
                candidates.extend(dataset_candidates)
                seen_audio_roots.add(resolved)
    return candidates


def mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = mono_waveform(waveform)
    return waveform, sample_rate


def check_audio_runtime(candidates: list[Candidate]) -> None:
    for candidate in candidates:
        try:
            load_audio(candidate.audio_path)
            return
        except Exception as exc:
            message = str(exc)
            if "TorchCodec is required" in message or "torchcodec" in message.lower():
                raise RuntimeError(
                    "torchaudio.load requires torchcodec in this environment. "
                    "Install dependencies with: pip install -r requirements.txt"
                ) from exc

    raise RuntimeError("Could not load any parsed audio sample with torchaudio.load.")


def cut_segment(candidate: Candidate, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if candidate.start_sec is not None and candidate.end_sec is not None:
        start_frame = max(0, int(round(candidate.start_sec * sample_rate)))
        end_frame = min(waveform.size(1), int(round(candidate.end_sec * sample_rate)))
        return waveform[:, start_frame:end_frame]
    return waveform


def resample_audio(waveform: torch.Tensor, original_sr: int, target_sr: int) -> torch.Tensor:
    if original_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, original_sr, target_sr)


def candidate_priority(candidate: Candidate) -> tuple[int, str]:
    if candidate.start_sec is None or candidate.end_sec is None:
        return (1, candidate.utt_id)
    duration = max(0.0, candidate.end_sec - candidate.start_sec)
    bucket = bucket_for_duration(duration)
    priority = {"long": 0, "mid": 1, "short": 2}[bucket]
    return (priority, candidate.utt_id)


def ordered_candidates(candidates: list[Candidate], seed: int) -> list[Candidate]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_dataset[candidate.dataset].append(candidate)

    ordered: list[Candidate] = []
    for dataset in DATASETS:
        items = by_dataset.get(dataset, [])
        rng.shuffle(items)
        items.sort(key=candidate_priority)
        ordered.extend(items)
    return ordered


def process_candidates(
    candidates: list[Candidate],
    output_root: Path,
    sample_rate: int,
    max_hours_per_corpus: float | None,
    seed: int,
) -> tuple[list[dict], Counter]:
    wav_dir = output_root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    dropped_path = output_root / "dropped_samples.jsonl"
    max_seconds = None if max_hours_per_corpus is None else max_hours_per_corpus * 3600.0
    used_seconds_by_dataset: dict[str, float] = defaultdict(float)
    completed_datasets: set[str] = set()
    manifest_rows: list[dict] = []
    dropped_reasons: Counter = Counter()
    cached_audio_path: Path | None = None
    cached_waveform: torch.Tensor | None = None
    cached_sample_rate: int | None = None

    with dropped_path.open("w", encoding="utf-8") as dropped_handle:
        for candidate in tqdm(ordered_candidates(candidates, seed), desc="Preparing samples"):
            if candidate.dataset in completed_datasets:
                continue

            text_raw = candidate.text_raw
            text = normalize_text(text_raw)
            drop_base = {
                "id": candidate.utt_id,
                "dataset": candidate.dataset,
                "source_audio": str(candidate.audio_path),
                "source_text": str(candidate.source_text),
            }

            if not text_raw.strip():
                record_drop(dropped_handle, dropped_reasons, drop_base, "empty_transcript")
                continue
            if len(text) <= 2:
                record_drop(dropped_handle, dropped_reasons, drop_base, "normalized_text_too_short")
                continue

            if cached_audio_path != candidate.audio_path:
                try:
                    cached_waveform, cached_sample_rate = load_audio(candidate.audio_path)
                    cached_audio_path = candidate.audio_path
                except Exception as exc:
                    record_drop(
                        dropped_handle,
                        dropped_reasons,
                        drop_base,
                        "torchaudio_load_failed",
                        error=str(exc),
                    )
                    cached_audio_path = None
                    cached_waveform = None
                    cached_sample_rate = None
                    continue

            if cached_waveform is None or cached_sample_rate is None:
                record_drop(dropped_handle, dropped_reasons, drop_base, "audio_cache_empty")
                continue

            waveform = cut_segment(candidate, cached_waveform, cached_sample_rate)
            try:
                waveform = resample_audio(waveform, cached_sample_rate, sample_rate)
            except Exception as exc:
                record_drop(
                    dropped_handle,
                    dropped_reasons,
                    drop_base,
                    "torchaudio_resample_failed",
                    error=str(exc),
                )
                continue

            duration = waveform.size(1) / float(sample_rate)

            if duration <= 1.0:
                record_drop(dropped_handle, dropped_reasons, drop_base, "duration_too_short", duration=duration)
                continue
            if max_seconds is not None and used_seconds_by_dataset[candidate.dataset] + duration > max_seconds:
                record_drop(
                    dropped_handle,
                    dropped_reasons,
                    drop_base,
                    "exceeds_max_hours_per_corpus",
                    duration=duration,
                )
                if max_seconds - used_seconds_by_dataset[candidate.dataset] <= 1.0:
                    completed_datasets.add(candidate.dataset)
                continue

            out_id = safe_id(candidate.dataset, candidate.utt_id)
            out_audio = wav_dir / f"{out_id}.wav"
            try:
                torchaudio.save(out_audio, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
            except Exception as exc:
                record_drop(
                    dropped_handle,
                    dropped_reasons,
                    drop_base,
                    "torchaudio_save_failed",
                    error=str(exc),
                )
                continue

            used_seconds_by_dataset[candidate.dataset] += duration
            if max_seconds is not None and max_seconds - used_seconds_by_dataset[candidate.dataset] <= 1.0:
                completed_datasets.add(candidate.dataset)

            manifest_rows.append(
                {
                    "id": out_id,
                    "audio": str(out_audio),
                    "text": text,
                    "text_raw": text_raw,
                    "dataset": candidate.dataset,
                    "duration": round(duration, 6),
                    "bucket": bucket_for_duration(duration),
                    "speaker": candidate.speaker,
                    "split": candidate.split,
                    "source_audio": str(candidate.audio_path),
                    "source_text": str(candidate.source_text),
                }
            )

    return manifest_rows, dropped_reasons


def write_manifest(output_root: Path, rows: list[dict]) -> Path:
    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_line(handle, row)
    return manifest_path


def write_summary(output_root: Path, rows: list[dict], dropped_reasons: Counter, manifest_path: Path) -> None:
    per_dataset = {
        dataset: {"samples": 0, "duration_hours": 0.0}
        for dataset in DATASETS
    }
    per_bucket = {
        bucket: {"samples": 0, "duration_hours": 0.0}
        for bucket in BUCKETS
    }
    for row in rows:
        duration_hours = row["duration"] / 3600.0
        per_dataset[row["dataset"]]["samples"] += 1
        per_dataset[row["dataset"]]["duration_hours"] += duration_hours
        per_bucket[row["bucket"]]["samples"] += 1
        per_bucket[row["bucket"]]["duration_hours"] += duration_hours

    for group in (per_dataset, per_bucket):
        for stats in group.values():
            stats["duration_hours"] = round(stats["duration_hours"], 6)

    summary = {
        "total_samples": len(rows),
        "total_duration_hours": round(sum(row["duration"] for row in rows) / 3600.0, 6),
        "per_dataset": per_dataset,
        "per_bucket": per_bucket,
        "dropped_count_by_reason": dict(sorted(dropped_reasons.items())),
        "manifest_path": str(manifest_path),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_kaldi(output_root: Path, rows: list[dict]) -> None:
    kaldi_dir = output_root / "kaldi"
    kaldi_dir.mkdir(parents=True, exist_ok=True)
    with (kaldi_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp, (
        kaldi_dir / "text"
    ).open("w", encoding="utf-8") as text_file, (kaldi_dir / "utt2spk").open(
        "w", encoding="utf-8"
    ) as utt2spk:
        for row in rows:
            wav_scp.write(f"{row['id']} {row['audio']}\n")
            text_file.write(f"{row['id']} {row['text']}\n")
            utt2spk.write(f"{row['id']} {row['speaker'] or row['id']}\n")


def prepare_output_dir(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    setup_logging()

    corpus_dirs = discover_corpus_dirs(args.data_root)
    for dataset, dirs in corpus_dirs.items():
        LOGGER.info("Discovered %s dirs: %s", dataset, ", ".join(str(path) for path in dirs) or "none")

    candidates = parse_candidates(corpus_dirs)
    if not candidates:
        raise RuntimeError(f"No parseable corpus samples found under {args.data_root}")

    check_audio_runtime(candidates)
    prepare_output_dir(args.output_root, args.overwrite)

    rows, dropped_reasons = process_candidates(
        candidates=candidates,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        max_hours_per_corpus=args.max_hours_per_corpus,
        seed=args.seed,
    )
    manifest_path = write_manifest(args.output_root, rows)
    write_summary(args.output_root, rows, dropped_reasons, manifest_path)
    if args.write_kaldi:
        write_kaldi(args.output_root, rows)

    LOGGER.info("Wrote %d manifest rows to %s", len(rows), manifest_path)
    LOGGER.info("Wrote summary to %s", args.output_root / "summary.json")


if __name__ == "__main__":
    main()
