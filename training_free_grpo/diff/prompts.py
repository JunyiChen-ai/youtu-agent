PROBLEM_WITH_EXPERIENCE_TEMPLATE = """You are answering multiple choice questions. Each question has options A, B, or C.

Experience snippets (if any) that may be helpful:
{experiences}

Question:
{problem}

Respond using exactly the following two lines and nothing else:
Thinking: <brief reasoning in one sentence>
Answer: <one letter, either A, B, or C>
"""


PROBLEM_WITHOUT_EXPERIENCE_TEMPLATE = """You are answering multiple choice questions. Each question has options A, B, or C.

Question:
{problem}

Respond using exactly the following two lines and nothing else:
Thinking: <brief reasoning in one sentence>
Answer: <one letter, either A, B, or C>
"""

# =====================
# Experience pipeline prompts (replicates web workflow for diff domain)
# =====================

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_SP = """You are an AI assistant specialized in analyzing multiple-choice (A/B/C) question-solving trajectories.
Your task is to summarize the provided trajectory by extracting decision points, reasoning, and evidence that led to a final choice.

Instructions:
1. For each step: identify the action (reasoning or tool), the extracted key evidence, and its influence on narrowing down choices.
2. If the ground-truth answer is provided, use it only to highlight where the attempt diverged or succeeded.
3. Keep summaries concise but specific about signals that support or contradict each option.

Output a clear, structured summary of the trajectory.
"""

SINGLE_ROLLOUT_SUMMARY_TEMPLATE_UP = """Task: Answer the following MCQ (A/B/C).

Question:
{task}

Trajectory (messages list):
{trajectory}

Correct Answer (if available): {answer}

Please produce a step-by-step summary as instructed.
"""

SINGLE_QUERY_CRITIQUE_TEMPLATE_SP = """You will review multiple attempts at the same MCQ problem. Your job is to extract generalizable experiences (concise guidelines) that help choose the correct option.

Guidelines:
- Focus on why certain signals/evidence support one option over others.
- Identify recurring reasoning mistakes and how to avoid them.
- Keep experiences concise, general, and applicable to similar MCQs.

Output Requirements:
Return raw text where the block between <Experiences> ... </Experiences> lists candidate experience snippets (one per line).
"""

SINGLE_QUERY_CRITIQUE_TEMPLATE_UP = """Question:
{question}

Correct Answer (if available): {answer}

Attempts and their summaries:
{attempts}

Please analyze the attempts and then output:
<Experiences>
- [One-line general experience 1]
- [One-line general experience 2]
...
</Experiences>
"""

GROUP_EXPERIENCE_UPDATE_TEMPLATE_SP = """You will reconcile new candidate experiences with the existing experience pool.

Given:
- Existing experiences (ID -> content)
- New candidate experiences (text lines)

For each candidate, decide an operation among ADD, UPDATE, DELETE, NONE.
Return a JSON array where each object has: {"operation": "ADD|UPDATE|DELETE|NONE", "id": "existing_id_or_null", "content": "..."}.
Only output the JSON array in a Markdown JSON block.
"""

GROUP_EXPERIENCE_UPDATE_TEMPLATE_UP = """Existing experiences:
{existing_experiences}

New candidate experiences:
{new_experiences}

Please return the JSON list of operations as specified.
"""

BATCH_EXPERIENCE_UPDATE_TEMPLATE_SP = """You will consolidate a batch of proposed operations into final updates for the experience pool.

Rules:
- Merge similar ADDs into one concise, general experience.
- If both UPDATE and DELETE target the same ID, prefer DELETE.
- Keep experiences short (one sentence) and broadly applicable.

Output a JSON array with objects of the form {"operation": "ADD|UPDATE|DELETE|NONE", "id": "id_or_null", "content": "..."}.
Only output the JSON array in a Markdown JSON block.
"""

BATCH_EXPERIENCE_UPDATE_TEMPLATE_UP = """Existing experiences and related operations:
{experiences_and_operations}

Produce the final consolidated decisions as the required JSON array.
"""
