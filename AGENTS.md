# Repo AI Instructions

These rules apply when creating or editing blog posts in this repository, especially under `src/data/blog/**`.

## Blog Tag Policy

- Treat tags as retrieval keys, not summaries.
- Use `2` to `4` tags per post.
- Use English `Title Case`.
- Prefer stable topic nouns over vague process words.
- Do not mix bilingual duplicates or near-synonyms.

## Tag Order

1. Domain tag first.
2. One primary topic tag.
3. Zero to two secondary topic tags only when they are real focuses of the post.
4. Add a context tag only if it is central to the note.

## Photography Tag Rules

- Any post inside `src/data/blog/photography/**` must include `Photography`.
- Pick the remaining tags from this controlled vocabulary whenever possible:
  - `Exposure`
  - `Metering`
  - `Focus`
  - `Depth of Field`
  - `Lens`
  - `Lighting`
  - `Flash`
  - `White Balance`
  - `Histogram`
  - `Dynamic Range`
  - `Color Grading`
  - `Color Theory`
  - `Curves`
  - `HSL`
  - `Masking`
  - `Workflow`
  - `Portrait`
  - `Landscape`
  - `Night`
  - `Low Light`

## Tag Selection Algorithm

1. Identify the post's single main retrieval intent.
2. Add the domain tag first.
3. Choose exactly one primary topic tag.
4. Add up to two secondary tags only if each one covers a meaningful part of the post or would realistically be used to find it later.
5. Stop at four tags even if the post touches more topics.

## Normalization Rules

- Prefer singular forms where possible.
- Do not add filler tags such as `Notes`, `Guide`, `Basics`, `Learning`, or `Tips`.
- Do not tag every subsection in a broad note. Tag the post's center of gravity.
- If unsure between a narrow and broad term, prefer the broader stable term already in the controlled vocabulary.

## Frontmatter Consistency

- File names should be English `kebab-case` and reflect the main topic.
- Titles can be Chinese and should stay concise.
- `description` should be one short sentence with no filler phrasing.

## Examples

- Exposure / metering / lighting note:
  - `Photography`
  - `Exposure`
  - `Metering`
  - `Lighting`
- Color grading / channel curves note:
  - `Photography`
  - `Color Grading`
  - `Color Theory`
  - `Curves`
