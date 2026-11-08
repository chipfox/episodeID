"""The golden checker must fail a plan that would mislead a human, and pass one
that defers exactly the undecidable files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import golden_check as G  # noqa: E402

SWAP = "Season 05/The X-Files - S05E08 - Kitsunegari - Bluray-1080p x265 Opus 5.1.mkv"
PLAIN = "Season 05/The X-Files - S05E12 - Bad Blood - Bluray-1080p x265 Opus 5.1.mkv"
UNDECIDABLE = "Season 05/The X-Files - S05E15 - Travelers - HDTV-1080p Proper AV1 Opus 5.1.mkv"

golden = {"swaps": {SWAP: "S05E10"}, "undecidable": [UNDECIDABLE]}


def item(kind, source, target=None):
    return {"kind": kind, "source": source, "reason": "test",
            "targets": [{"path": target}] if target else []}


swap_rename = item("rename", SWAP, "Season 05/x/The X-Files - S05E10 - Chinga - Bluray-1080p x265 Opus 5.1.mkv")
plain_ok = item("ok", PLAIN)
undecided_review = item("review", UNDECIDABLE)

good = G.score({"items": [swap_rename, plain_ok, undecided_review]}, golden)
assert good["correct"] == 2, good
assert good["deferred_correctly"] == [UNDECIDABLE], good
assert not (good["wrong_ok"] or good["false_renames"] or good["missed_swaps"]), good

left_alone = G.score({"items": [item("ok", SWAP), plain_ok]}, golden)
assert left_alone["wrong_ok"] == [SWAP], left_alone

wrong_swap = G.score({"items": [item("rename", SWAP,
                                     "Season 05/x/The X-Files - S05E09 - Schizogeny - Bluray-1080p x265 Opus 5.1.mkv"),
                                plain_ok]}, golden)
assert wrong_swap["missed_swaps"] == [SWAP], wrong_swap

renamed_truth = G.score({"items": [swap_rename,
                                   item("rename", PLAIN,
                                        "Season 05/x/The X-Files - S05E13 - Irresistible - Bluray-1080p x265 Opus 5.1.mkv"),
                                   undecided_review]}, golden)
assert renamed_truth["false_renames"][0]["to"] == "S05E13", renamed_truth

deferred_known = G.score({"items": [swap_rename, item("review", PLAIN), undecided_review]}, golden)
assert deferred_known["deferred_wrongly"] == [PLAIN], deferred_known

print("golden_check: OK")
