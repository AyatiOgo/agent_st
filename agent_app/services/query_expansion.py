
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini",
)


def expand_query(question, num_queries=3):
    """
    Generate alternative search queries while preserving
    the student's original intent.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You improve search queries for a university "
                    "admissions and student handbook. Generate "
                    "alternative queries that preserve the user's "
                    "original intent while using different relevant "
                    "terms and phrasing. Do not answer the question. "
                    "Do not introduce new assumptions. Return only "
                    "a valid JSON array of strings."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n"
                    f"Generate {num_queries} alternative search queries."
                ),
            },
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content

    # Remove Markdown code fences if the model includes them
    content = re.sub(
    r"^```(?:json)?\s*|\s*```$",
    "",
    content.strip(),
    flags=re.IGNORECASE,
)


    try:
        queries = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return [question]

    if not isinstance(queries, list):
        return [question]

    # Keep only valid, unique query strings.
    expanded = [question]
    seen = {question.strip().casefold()}

    for query in queries:
        if not isinstance(query, str):
            continue

        query = query.strip()

        if query and query.casefold() not in seen:
            expanded.append(query)
            seen.add(query.casefold())

        if len(expanded) >= num_queries + 1:
            break

    return expanded
