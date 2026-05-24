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
    action_text = {}
    for trajectory in trajectories:
        seen_in_trajectory = set()
        for action in trajectory["actions"]:
            normalized = normalize_action(action)
            if not normalized or normalized in seen_in_trajectory:
                continue
            seen_in_trajectory.add(normalized)
            action_counts[normalized] += 1
            action_text.setdefault(normalized, action)

    threshold = max(min_vote_count, int(len(trajectories) * min_vote_ratio + 0.999))
    selected = [
        (normalized, count)
        for normalized, count in action_counts.items()
        if count >= threshold
    ]
    selected.sort(key=lambda item: (-item[1], item[0]))
    selected = selected[:max_reference_steps]
    return [f"Search {action_text[normalized]}" for normalized, _ in selected]


def build_vote_request(key, group, max_actions_per_group=24):
    action_counts = Counter()
    for trajectory in group["trajectories"]:
        for action in trajectory["actions"]:
            normalized = normalize_action(action)
            if normalized:
                action_counts[action] += 1
    ranked_actions = [
        {"action": action, "count": count}
        for action, count in action_counts.most_common(max_actions_per_group)
    ]
    metadata = group["metadata"]
    user_prompt = {
        "question": metadata.get("question"),
        "candidate_actions": ranked_actions,
        "instruction": (
            "Select the minimal consensus search-intent steps needed to solve the question. "
            "Return JSON only with key reference_steps. Each step must be a short 'Search ...' intent. "
            "Do not include fallback branches, exhaustive lists, URLs, or trajectory tags."
        ),
    }
    return {
        "custom_id": "|".join(key),
        "messages": [
            {
                "role": "system",
                "content": "You generate concise reference search plans for QA reasoning trajectories.",
            },
            {
                "role": "user",
                "content": json.dumps(user_prompt, ensure_ascii=False),
            },
        ],
        "metadata": metadata,
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
