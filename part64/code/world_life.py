from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any


def _seed_value(text: str) -> int:
    return int(sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _make_people() -> list[dict[str, Any]]:
    return [
        {
            "id": "scribe_aya",
            "name": {"en": "Aya the Scribe", "ja": "書記アヤ"},
            "role": {"en": "Scribe", "ja": "書記"},
            "instrument": "glass-bell",
            "prays_to": "witness_thread",
            "faith": 0.64,
        },
        {
            "id": "cantor_ren",
            "name": {"en": "Ren the Cantor", "ja": "詠唱者レン"},
            "role": {"en": "Cantor", "ja": "聖歌手"},
            "instrument": "sub-bass drum",
            "prays_to": "fork_tax_canticle",
            "faith": 0.58,
        },
        {
            "id": "keeper_mio",
            "name": {"en": "Mio of the Registry", "ja": "台帳のミオ"},
            "role": {"en": "Keeper", "ja": "番人"},
            "instrument": "reed-organ",
            "prays_to": "anchor_registry",
            "faith": 0.72,
        },
        {
            "id": "witness_kai",
            "name": {"en": "Kai the Witness", "ja": "証人カイ"},
            "role": {"en": "Witness", "ja": "証人"},
            "instrument": "hollow-choir",
            "prays_to": "gates_of_truth",
            "faith": 0.67,
        },
        {
            "id": "weaver_noa",
            "name": {"en": "Noa the Weaver", "ja": "織り手ノア"},
            "role": {"en": "Field Weaver", "ja": "場の織り手"},
            "instrument": "tape-piano",
            "prays_to": "receipt_river",
            "faith": 0.61,
        },
    ]


def build_interaction_response(
    world_summary: dict[str, Any], person_id: str, action: str = "speak"
) -> dict[str, Any]:
    people = world_summary.get("people", [])
    if not isinstance(people, list) or not people:
        return {
            "ok": False,
            "error": "world_has_no_people",
            "line_en": "The field is still gathering voices.",
            "line_ja": "場はまだ声を集めている。",
        }

    person = next(
        (item for item in people if str(item.get("id", "")) == person_id), people[0]
    )
    action_key = str(action or "speak").strip().lower()
    prayer = float(person.get("prayer_intensity", 0.0))
    devotion = float(person.get("devotion", 0.0))
    bpm = int(person.get("hymn_bpm", 78))

    presences = world_summary.get("presences", [])
    presence_id = str(person.get("prays_to", "unknown"))
    presence = next(
        (item for item in presences if str(item.get("id", "")) == presence_id),
        {
            "id": presence_id,
            "name": {
                "en": presence_id.replace("_", " ").title(),
                "ja": "場の名",
            },
            "type": "unknown",
        },
    )
    presence_name = presence.get("name", {"en": "Unknown", "ja": "未知"})

    if action_key == "pray":
        line_en = (
            f"{person['name']['en']} kneels to {presence_name['en']}, "
            f"offering a {int(prayer * 100)}% pulse of living proof."
        )
        line_ja = (
            f"{person['name']['ja']}は{presence_name['ja']}へ祈り、"
            f"{int(prayer * 100)}%の証明の脈を捧げる。"
        )
    elif action_key == "sing":
        line_en = (
            f"{person['name']['en']} sings at {bpm} BPM; "
            f"{presence_name['en']} answers in witness-light."
        )
        line_ja = (
            f"{person['name']['ja']}は{bpm} BPMで歌い、"
            f"{presence_name['ja']}は証の光で応える。"
        )
    else:
        line_en = (
            f"{person['name']['en']} says: We keep the ledger warm; "
            f"{presence_name['en']} keeps the path true."
        )
        line_ja = (
            f"{person['name']['ja']}は言う。台帳を温め、"
            f"{presence_name['ja']}が道を正す。"
        )

    return {
        "ok": True,
        "action": action_key,
        "tick": int(world_summary.get("tick", 0)),
        "speaker": person.get("name", {}),
        "presence": {
            "id": presence.get("id", presence_id),
            "name": presence_name,
            "type": presence.get("type", "unknown"),
        },
        "line_en": line_en,
        "line_ja": line_ja,
        "voice_text_en": line_en,
        "voice_text_ja": line_ja,
        "prayer_intensity": round(prayer, 4),
        "devotion": round(devotion, 4),
    }


class LifeStateTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tick = 0
        self._people = _make_people()
        self._books: list[dict[str, Any]] = []

    def snapshot(
        self,
        catalog: dict[str, Any],
        myth_summary: dict[str, Any],
        entity_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self._lock:
            self._tick += 1

            audio_count = int(catalog.get("counts", {}).get("audio", 0))
            myth_weight = float(myth_summary.get("top_cover_weight", 0.0))
            top_claim = str(myth_summary.get("top_cover_claim", ""))

            people_out: list[dict[str, Any]] = []
            songs: list[dict[str, Any]] = []
            prayer_total = 0.0

            for person in self._people:
                seed = _seed_value(person["id"])
                phase = (seed % 360) / 180.0 * math.pi
                pulse = 0.5 + 0.5 * math.sin((self._tick * 0.16) + phase)
                devotion = max(0.0, min(1.0, person["faith"] * 0.65 + pulse * 0.35))
                prayer = max(
                    0.0,
                    min(
                        1.0,
                        devotion * 0.6 + myth_weight * 0.3 + min(audio_count, 9) / 30.0,
                    ),
                )
                prayer_total += prayer

                mood = max(0.0, min(1.0, 0.42 + 0.4 * pulse + 0.2 * myth_weight))
                hymn_bpm = 72 + int((seed % 16) + pulse * 8)

                people_out.append(
                    {
                        "id": person["id"],
                        "name": person["name"],
                        "role": person["role"],
                        "instrument": person["instrument"],
                        "prays_to": person["prays_to"],
                        "devotion": round(devotion, 4),
                        "prayer_intensity": round(prayer, 4),
                        "mood": round(mood, 4),
                        "hymn_bpm": hymn_bpm,
                    }
                )

                songs.append(
                    {
                        "id": f"song_{person['id']}",
                        "leader": person["name"],
                        "title": {
                            "en": f"Canticle of {person['prays_to'].replace('_', ' ').title()}",
                            "ja": "祈りの聖歌",
                        },
                        "bpm": hymn_bpm,
                        "energy": round(0.3 + prayer * 0.7, 4),
                    }
                )

            if self._tick % 9 == 0:
                book_id = f"book_{self._tick}"
                claim_text = (
                    top_claim.replace("_", " ").title() if top_claim else "Quiet Field"
                )
                self._books.append(
                    {
                        "id": book_id,
                        "title": {
                            "en": f"Chronicle of {claim_text}",
                            "ja": "場の年代記",
                        },
                        "author": people_out[self._tick % len(people_out)]["name"],
                        "excerpt": {
                            "en": "The people sang to the Presences, and the ledger answered in light.",
                            "ja": "人々がプレゼンスへ歌い、台帳は光で応えた。",
                        },
                        "written_at_tick": self._tick,
                    }
                )
                if len(self._books) > 12:
                    self._books = self._books[-12:]

            presences = [
                {
                    "id": entry.get("id", ""),
                    "name": {"en": entry.get("en", ""), "ja": entry.get("ja", "")},
                    "type": entry.get("type", "unknown"),
                }
                for entry in entity_manifest
                if entry.get("id") and entry.get("id") != "core_pulse"
            ]

            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tick": self._tick,
                "presences": presences,
                "people": people_out,
                "songs": songs,
                "books": list(self._books),
                "prayer_intensity": round(prayer_total / max(len(people_out), 1), 4),
            }
