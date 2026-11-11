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

# ---------------------------------------------------------------- containment
# "Does this reference contain the dialogue we heard" — the measure the transcript
# view uses. Its point is that a short sample of a text is *fully* contained in the
# long text it was cut from, which is what a cosine gets wrong.
import random  # noqa: E402

VOCAB = [f"{a}{b}" for a in "bcdfghjklmnprstvwz" for b in "aeiou"]


def transcript(n, markers, seed):
    """Varied dialogue-like text: unique word pairs, with the given markers recurring."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.extend(markers[:2] if i % 5 == 0 else [rng.choice(VOCAB)])
    return " ".join(out)


sample = transcript(200, ("spaceship", "alaska"), 7)          # ten minutes of an episode
episode = sample + " " + transcript(150, ("mulder", "scully"), 8)   # its full reference

assert abs(textsim.containment(textsim.bigrams(sample), textsim.bigrams(episode)) - 1.0) < 1e-9
assert textsim.containment(textsim.bigrams(episode), textsim.bigrams(sample)) < 1.0
assert textsim.containment(textsim.bigrams(sample), textsim.bigrams(other)) < 0.1
assert len(textsim.bigrams(sample)) >= textsim.BIGRAM_FLOOR

# Length-robust where the cosine is not: the same sample scores the same against a
# reference twice as long, because containment divides by the query only.
longer = episode + " " + transcript(150, ("albatross", "harbor"), 9)
assert abs(textsim.containment(textsim.bigrams(sample), textsim.bigrams(episode)) -
           textsim.containment(textsim.bigrams(sample), textsim.bigrams(longer))) < 1e-9
assert textsim.similarity(sample, longer) < textsim.similarity(sample, episode), (
    "the cosine should punish the longer reference; containment should not")

# overlap() is the duplicate check's question: the larger share of either text.
assert abs(textsim.overlap(textsim.bigrams(episode), textsim.bigrams(sample)) - 1.0) < 1e-9
assert textsim.overlap(textsim.bigrams(a), textsim.bigrams(a)) == 1.0
assert textsim.containment(set(), textsim.bigrams(a)) == 0.0
assert textsim.overlap(set(), textsim.bigrams(a)) == 0.0

print("textsim: OK")
