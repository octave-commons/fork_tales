# Deep Research Runbook (Operation Mindfuck)

This is the **human-in-the-loop** protocol for “real deep research later” without getting seduced by pretty output.

## 0) Define the question as a *claim budget*
Write:
- Question:
- What counts as evidence here?
- What doesn’t?
- Max number of claims allowed in the final report (start small).

## 1) Freeze a snapshot (P)
Run the build:
- `build/manifest.jsonl` is the frozen file list.
- Keep the produced split zips as the snapshot artifacts.

## 2) Retrieve (R)
For each claim you want in the report:
- retrieve the smallest possible set of snippets that could support it.
- prefer primary artifacts (story text, notes, recorded decisions).

## 3) Normalize (N)
Convert each supporting snippet into:
- an S3 anchor line (claim_id, snippet, sha256).
- an S5 resolution reference (which entity IDs are involved).

## 4) Π (Graph / Model)
Only after anchors exist:
- create edges
- create forks
- create ideologies / faction models

**Rule:** Every edge has an anchor. No exceptions.

## 5) Act (A)
Write the report using:
- claims that point to anchors
- forks that cite intent events
- entity references that cite S5

## 6) Feedback loop
When someone disputes a claim:
- check the anchor
- if anchor weak, downgrade claim
- if entity unclear, open an S5 change request
- if fork noisy, raise the intent gate

## “Default voice” failure mode
If the model starts explaining why it followed constraints:
- treat that as *unanchored narration*.
- redirect to: “show the anchors, show the data, show the artifacts.”

## Deliverable templates
- `report_template.md` (in this folder)
- `claim_card.md` (in this folder)


## Anti-meta output contract
When the model starts narrating why it complied:
- treat as *unanchored justification*
- require it to output one of: (a) anchors, (b) data artifact, (c) new chapter/song
- any explanation must cite file paths + claim IDs
