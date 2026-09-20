#!/usr/bin/env python3
"""Prompt construction for every arm.

Section 4 requires the same prompt content for every LLM arm: same
instructions, same category descriptions. Only the final output instruction
differs, because the mechanisms differ. Keeping one shared body in one place is
what makes that claim true rather than aspirational.

Frozen by commit before the evaluation run.
"""

import taxonomy
import protocol as P

LETTERS = "ABCDEFG"

SHARED_BODY_MARKER = "Which arXiv primary category did the authors submit"

_TASK = (
    "You are given the title and abstract of an arXiv paper.\n\n"
    f"{SHARED_BODY_MARKER} this paper under?\n\n"
    "The primary category is the single category the submitting authors chose "
    "first. Papers are often cross-listed, but exactly one category is the "
    "primary one.\n"
)

# Section 4: goes to every arm for symmetry. Not an anti-reasoning instruction.
_TAGS = "Do not include internal or system XML tags in your response."

# Arm J's `instructions` field. Derived from the SAME _TASK text the chat arms
# receive, plus section 4's universal XML-tag line, so section 4's "same prompt
# content" claim holds across the transport boundary rather than relying on two
# hand-kept copies agreeing. J emits no free text, so the tag line is inert for
# it -- it is present because section 4 says the instruction goes to every arm.
JEV_INSTRUCTIONS = f"{_TASK}\n{_TAGS}"

# Module-level cache for descriptions. Section 3 requires them verbatim,
# so they are read from the committed file once and reused.
_DESCRIPTIONS = None


def _descriptions():
    """Load the frozen descriptions once. Section 3 requires them verbatim,
    so they are read from the committed file, never rebuilt per call.
    """
    global _DESCRIPTIONS
    if _DESCRIPTIONS is None:
        _DESCRIPTIONS = taxonomy.load_descriptions()
    return _DESCRIPTIONS


def _options_block(options):
    desc = _descriptions()
    lines = []
    for i, label in enumerate(options):
        e = desc[label]
        lines.append(f"{LETTERS[i]}. {label} ({e['name']}): {e['description']}")
    return "\n".join(lines)


def _body(item):
    return (
        f"{_TASK}\n"
        f"Candidate categories:\n{_options_block(item['options'])}\n\n"
        f"Title: {item['title']}\n\n"
        f"Abstract: {item['abstract']}\n"
    )


def json_schema(options):
    """Schema requiring all 7 keys, presented in this item's shuffled order.

    Section 4 puts structured outputs on every verbalized arm so that parse
    failures cannot differ across arms for reasons unrelated to calibration.
    """
    return {
        "name": "category_distribution",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                label: {
                    "type": "number",
                    "description": f"Probability the primary category is {label}",
                } for label in options
            },
            "required": list(options),
            "additionalProperties": False,
        },
    }


def build_prompt(item, mechanism):
    """Messages for one item under one mechanism."""
    if mechanism == "verbalized":
        tail = (
            "Respond with a JSON object giving your probability for each of the "
            "seven categories. Use the exact category keys shown above. The "
            "seven values must sum to 1."
        )
    elif mechanism == "letter":
        tail = (
            f"Respond with exactly one letter, {LETTERS[0]} through "
            f"{LETTERS[-1]}, naming the category you believe is the primary "
            "one. Output only that single letter and nothing else."
        )
    elif mechanism == "decisions":
        # J returns a typed choice; the option list is the choice set. The
        # transport shape is settled by the Task 7 probe, not here.
        tail = (
            "Choose the category you believe is the primary one."
        )
    else:
        raise ValueError(f"unknown mechanism: {mechanism!r}")

    return [{"role": "user", "content": f"{_body(item)}\n{tail}\n\n{_TAGS}"}]
