from typing import Any

import mlflow
import numpy as np

from thesis_graph.embedding import embed_text


def get_researchers_features(
    researchers: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    names = []
    descriptions = []

    for researcher in researchers:
        names.append(researcher["name"])
        interests = researcher.get("interests", [])
        # articles = researcher.get("articles", [])

        desc = f"Interests of research: {','.join(interests)}"
        descriptions.append(desc)

    mlflow.log_param("researchers_features", "Interests")
    embeddings = embed_text(descriptions)
    return {name: embedding for name, embedding in zip(names, embeddings)}


def build_mentors_features_matrix(
    researchers: list[dict[str, Any]], mentors_dict: dict[str, int]
):
    features = get_researchers_features(researchers)

    if len(features) != len(mentors_dict):
        raise ValueError("Some mentors are missing in researchers features.")

    matrix = np.zeros(
        (len(mentors_dict), next(iter(features.values())).shape[0]), dtype=np.float32
    )

    for mentor_name, index in mentors_dict.items():
        matrix[index] = features[mentor_name]

    return matrix
