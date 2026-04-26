from __future__ import annotations

from pathlib import Path

from trs.domain.schemas import LibrarySnapshot, SkillCard
from trs.storage.jsonl_store import read_jsonl, write_jsonl
from trs.storage.manifest import write_manifest
from trs.utils.ids import make_record_id


def save_skill_cards(path: str | Path, cards: list[SkillCard]) -> None:
    write_jsonl(path, [card.to_dict() for card in cards])


def load_skill_cards(path: str | Path) -> list[SkillCard]:
    return [SkillCard.from_dict(row) for row in read_jsonl(path)]


def build_library_snapshot(cards: list[SkillCard], output_dir: str | Path, profile: dict[str, object]) -> LibrarySnapshot:
    if not cards:
        raise ValueError("Cannot build a library snapshot from zero skill cards.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cards_file = output / "cards.jsonl"
    manifest_file = output / "library_manifest.json"
    write_jsonl(cards_file, [card.to_dict() for card in cards])
    task_type = cards[0].task_type
    dataset = cards[0].dataset
    snapshot = LibrarySnapshot(
        id=make_record_id("library", f"{dataset}:{profile.get('name', 'profile')}:{len(cards)}"),
        task_type=task_type,
        dataset=dataset,
        retriever_type=str(profile.get("retriever_type", "bm25")),
        cards_file=str(cards_file),
        metadata_file=str(manifest_file),
        metadata={"profile": profile, "card_count": len(cards)},
    )
    write_manifest(manifest_file, snapshot.to_dict())
    return snapshot
