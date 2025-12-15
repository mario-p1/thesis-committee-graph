from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from thesis_graph.data import load_thesis_csv, train_test_split_thesis_df
from thesis_graph.embedding import embed_text


def build_single_graph(
    thesis_df: pd.DataFrame,
    mentors_dict: dict[str, int],
    researchers_df: pd.DataFrame | None = None,
) -> HeteroData:
    # Build Mentors features

    # researchers_features = build_mentors_features_matrix(researchers_df, mentors_dict)
    # researchers_features = torch.from_numpy(researchers_features)

    # Build thesis features
    desc_embeddings = embed_text(thesis_df["thesis_desc_en"].tolist())
    thesis_features = torch.from_numpy(desc_embeddings)

    # Supervises relation
    supervises_mentor = thesis_df["mentor"].apply(lambda mentor: mentors_dict[mentor])
    supervises_thesis = thesis_df.index.tolist()
    supervises_features = torch.vstack(
        [
            torch.LongTensor(supervises_thesis),
            torch.LongTensor(supervises_mentor),
        ]
    )

    # Build graph
    graph = HeteroData()
    graph["thesis"].node_id = torch.arange(len(thesis_df))
    graph["thesis"].x = thesis_features

    graph["mentor"].node_id = torch.tensor(list(mentors_dict.values()))
    # graph["mentor"].x = researchers_features

    graph["thesis", "supervised_by", "mentor"].edge_index = supervises_features
    graph["mentor", "supervises", "thesis"].edge_index = supervises_features.flip(0)

    # Validate the constructed graph
    validate_result = graph.validate()
    if not validate_result:
        raise Exception("Constructed graph is not valid")

    return graph


def build_mentors_dict(thesis_df: pd.DataFrame) -> dict[str, int]:
    mentors = sorted(thesis_df["mentor"].unique().tolist())
    mentors_dict = {mentor: index for index, mentor in enumerate(mentors)}
    return mentors_dict


def build_graphs(thesis_path: Path, train_ratio: float = 0.8, val_ratio: float = 0.1):
    thesis_df = load_thesis_csv(thesis_path)

    train_df, val_df, test_df = train_test_split_thesis_df(
        thesis_df, train_ratio=train_ratio, val_ratio=val_ratio
    )

    mentors_dict = build_mentors_dict(train_df)

    train_data = build_single_graph(train_df, mentors_dict=mentors_dict)
    val_data = build_single_graph(val_df, mentors_dict=mentors_dict)
    test_data = build_single_graph(test_df, mentors_dict=mentors_dict)

    return mentors_dict, train_data, val_data, test_data
