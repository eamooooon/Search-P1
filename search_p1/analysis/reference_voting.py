import json
from collections import Counter

from search_p1.analysis.reference_io import key_from_custom_id, read_jsonl, row_key
from search_p1.analysis.reward_format import qa_em_format


def normalize_action(action):
    return qa_em_format.normalize_step(action)


def build_consensus_reference_steps(group, min_vote_count=2, min_vote_ratio=0.2, max_reference_steps=4):
    trajectories = group["trajectories"]
    if not trajectories:
        return []

    action_counts = Counter()
    action_positions = {}
    action_text = {}
    for trajectory in trajectories:
        seen_in_trajectory = set()
        for position, action in enumerate(trajectory["actions"]):
            normalized = normalize_action(action)
            if not normalized or normalized in seen_in_trajectory:
                continue
            seen_in_trajectory.add(normalized)
            action_counts[normalized] += 1
            action_positions.setdefault(normalized, []).append(position)
            action_text.setdefault(normalized, action)

    threshold = max(min_vote_count, int(len(trajectories) * min_vote_ratio + 0.999))
    selected = [
        (normalized, count)
        for normalized, count in action_counts.items()
        if count >= threshold
    ]
    selected.sort(key=lambda item: (-item[1], item[0]))
    selected = selected[:max_reference_steps]
    selected.sort(key=lambda item: (
        sum(action_positions[item[0]]) / len(action_positions[item[0]]),
        item[1] * -1,
        item[0],
    ))
    return [f"Search {action_text[normalized]}" for normalized, _ in selected]


def _metadata_targets(metadata):
    ground_truth = metadata.get("ground_truth")
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target")
        if isinstance(target, list):
            return [str(item) for item in target]
        if target is not None:
            return [str(target)]
    return []


def _ranked_group_actions(group, max_actions=24):
    action_counts = Counter()
    for trajectory in group["trajectories"]:
        for action in trajectory["actions"]:
            normalized = normalize_action(action)
            if normalized:
                action_counts[action] += 1
    return [
        {"action": action, "count": count}
        for action, count in action_counts.most_common(max_actions)
    ]


def _paper_prompt(metadata, ranked_actions=None):
    question = metadata.get("question") or ""
    golden_answers = _metadata_targets(metadata)
    ranked_actions = ranked_actions or []
    if ranked_actions:
        candidate_text = "\n".join(
            f"- ({item['count']} successful trajectories) {item['action']}"
            for item in ranked_actions
        )
    else:
        candidate_text = "- No candidate searches were provided."
    return f"""You are an expert planner and reasoning optimizer.
Current Question: {question}
Correct Answer: {golden_answers}
Candidate search queries from successful trajectories:
{candidate_text}

Your task is to generate:
1. Optimized Reasoning Path: A sequence of Search steps that would lead directly to the correct
answer in the most efficient way. Format as a numbered list, and start every item with "Search ".
2. Optimized Planner: A concise, step-by-step instruction on how a reasoning agent should solve
this question correctly and efficiently.
Important:
• Focus on the minimal set of queries needed.
• Prefer concrete entity/relation search queries from the candidate list when they are useful.
• Do not output placeholders such as "query 1", "query 2", or generic Search steps.
• Avoid redundant or inefficient steps.
Output format:
<correct_reasoning_path>
1. Search concrete entity/relation query
2. Search concrete entity/relation query
</correct_reasoning_path>
<optimized_planner>
To solve this, first search for... then...
</optimized_planner>"""


def _json_prompt(metadata, ranked_actions):
    question = metadata.get("question") or ""
    golden_answers = _metadata_targets(metadata)
    candidate_text = "\n".join(
        f"{index}. ({item['count']} successful trajectories) {item['action']}"
        for index, item in enumerate(ranked_actions, start=1)
    )
    return f"""Question:
{question}

Correct answer:
{golden_answers}

Candidate search actions from successful trajectories:
{candidate_text}

Task:
Select the minimal set of concrete search steps needed to solve the question.
Use the candidate actions when they are useful, but merge duplicates and remove redundant steps.

Return exactly one JSON object with this schema:
{{"reference_steps": ["Search concrete entity/relation query", "Search concrete entity/relation query"]}}

Rules:
- The only top-level key must be "reference_steps".
- Every item must start with "Search ".
- Do not copy this prompt.
- Do not return keys like "question", "candidate_actions", or "instruction".
- Do not include explanations, markdown, fallback branches, URLs, or trajectory tags."""


def build_vote_request(key, group, max_actions_per_group=12, prompt_style="consensus_json"):
    metadata = group["metadata"]
    if prompt_style == "paper":
        ranked_actions = _ranked_group_actions(group, max_actions=max_actions_per_group)
        return {
            "custom_id": "|".join(key),
            "messages": [
                {
                    "role": "user",
                    "content": _paper_prompt(metadata, ranked_actions=ranked_actions),
                },
            ],
            "metadata": metadata,
            "prompt_style": prompt_style,
        }

    ranked_actions = _ranked_group_actions(group, max_actions=max_actions_per_group)
    return {
        "custom_id": "|".join(key),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You generate concise reference search plans for QA reasoning trajectories. "
                    "You must return exactly one JSON object and nothing else."
                ),
            },
            {
                "role": "user",
                "content": _json_prompt(metadata, ranked_actions),
            },
        ],
        "metadata": metadata,
        "prompt_style": prompt_style,
    }


def load_llm_votes(path):
    if not path:
        return {}
    votes = {}
    for row in read_jsonl(path):
        key = row_key(row) or key_from_custom_id(row.get("custom_id"))
        steps = row.get("reference_steps")
        if key is not None and steps:
            votes[key] = steps
    return votes


def build_reference_rows(groups,
                         llm_votes=None,
                         min_successful=1,
                         min_vote_count=2,
                         min_vote_ratio=0.2,
                         max_reference_steps=4,
                         return_stats=False):
    rows = []
    llm_votes = llm_votes or {}
    stats = {
        "eligible_groups": 0,
        "eligible_groups_without_reference": 0,
        "consensus_reference_rows": 0,
        "llm_reference_rows": 0,
    }
    for key, group in groups.items():
        if group["correct"] < min_successful or not group["trajectories"]:
            continue
        stats["eligible_groups"] += 1
        metadata = group["metadata"]
        voted_steps = llm_votes.get(key)
        if voted_steps:
            reference_steps = voted_steps[:max_reference_steps]
            source = "llm_vote"
        else:
            reference_steps = build_consensus_reference_steps(
                group,
                min_vote_count=min_vote_count,
                min_vote_ratio=min_vote_ratio,
                max_reference_steps=max_reference_steps,
            )
            source = "consensus"
        if not reference_steps:
            stats["eligible_groups_without_reference"] += 1
            continue
        if source == "llm_vote":
            stats["llm_reference_rows"] += 1
        else:
            stats["consensus_reference_rows"] += 1

        row = {
            "data_source": metadata.get("data_source"),
            "split": metadata.get("split"),
            "index": metadata.get("index"),
            "question": metadata.get("question"),
            "reference_steps": reference_steps,
            "reference_plan_source": source,
            "source_trajectory_count": group["total"],
            "accepted_trajectory_count": group["correct"],
        }
        rows.append({k: v for k, v in row.items() if v is not None})
    if return_stats:
        return rows, stats
    return rows
