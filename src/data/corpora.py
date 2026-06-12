import logging
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)
AUDIO_SUFFIXES = {".wav", ".flac"}
DATASETS = ("zeroth", "pansori_tedxkr", "asr_kcsc")
BUCKETS = ("short", "mid", "long")
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")


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
    original_audio_path: Path | None = None
    original_start_sec: float | None = None
    original_end_sec: float | None = None


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


def archive_root_for(data_root: Path) -> Path:
    download_root = data_root / "download"
    if download_root.is_dir():
        return download_root
    return data_root


def extracted_root_for(data_root: Path) -> Path:
    if data_root.name == "extracted":
        return data_root
    return archive_root_for(data_root) / "extracted"


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def extraction_target_name(archive_path: Path) -> str:
    name = archive_path.name
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return archive_path.stem


def extract_corpus_archives(data_root: Path) -> None:
    archive_root = archive_root_for(data_root)
    if not archive_root.exists() or archive_root.name == "extracted":
        return

    extract_root = archive_root / "extracted"
    for archive_path in sorted(archive_root.iterdir()):
        if archive_path.is_dir() or infer_corpus_key(archive_path) is None or not is_archive(archive_path):
            continue

        target_dir = extract_root / extraction_target_name(archive_path)
        marker = target_dir / ".extract_complete"
        if marker.exists():
            continue

        LOGGER.info("Extracting %s to %s", archive_path, target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if archive_path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zip_file:
                zip_file.extractall(target_dir)
        else:
            with tarfile.open(archive_path) as tar_file:
                tar_file.extractall(target_dir)
        marker.write_text("ok\n", encoding="utf-8")


def discover_corpus_dirs(data_root: Path) -> dict[str, list[Path]]:
    extract_corpus_archives(data_root)
    search_root = extracted_root_for(data_root)
    if not search_root.exists():
        search_root = data_root

    discovered: dict[str, list[Path]] = {dataset: [] for dataset in DATASETS}
    for path in sorted(search_root.rglob("*")):
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


def parse_candidates_from_data_root(data_root: Path) -> list[Candidate]:
    return parse_candidates(discover_corpus_dirs(data_root))
