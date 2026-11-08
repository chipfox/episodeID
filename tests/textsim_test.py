"""One comparison for "are these two transcripts the same", used by reference
verification, repair validation and conflict detection alike."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import textsim  # noqa: E402

a = ("Scully is diagnosed with terminal brain cancer and asks Mulder to stay "
     "with her through the treatment.")
b = ("Mulder, I have cancer. Statistically there is about zero chance of survival, "
     "which means the treatment is all we have left.")
other = "The Knicks won in overtime last night at the garden after a wild finish."

assert abs(textsim.similarity(a, a) - 1.0) < 1e-9, textsim.similarity(a, a)
assert abs(textsim.similarity(a, b) - textsim.similarity(b, a)) < 1e-12
assert textsim.similarity(a, other) < 0.2, textsim.similarity(a, other)

score = textsim.scorer([a, b])
assert abs(float(score(a)[0]) - 1.0) < 1e-9, score(a)
assert float(score(a)[1]) < 0.3, score(a)  # related subject matter, not the same scene

sims = textsim.pairwise([a, b, other])
assert sims.shape == (3, 3), sims.shape
assert all(abs(sims[i][i] - 1.0) < 1e-9 for i in range(3))
assert abs(sims[0][1] - sims[1][0]) < 1e-12
assert sims[0][2] < 0.2, sims[0][2]

# The two views other modules use — conflict detection (pairwise) and repair
# validation (references._text_similarity) — must produce the same number.
from app import references  # noqa: E402
assert abs(textsim.pairwise([a, b])[0][1] - references._text_similarity(a, b)) < 1e-12

print("textsim: OK")
