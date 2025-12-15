import serpapi
import tqdm

from data_builder.serpapi_client import get_serpapi_client
from thesis_graph.config import BASE_DATA_PATH
from thesis_graph.utils import get_current_time_str
from thesis_graph.utils import save_json_to_file
from thesis_graph.utils import load_json_file


def fetch_profile_details(client: serpapi.Client, author_id: str):
    search_params = {
        "engine": "google_scholar_author",
        "author_id": author_id,
        "hl": "en",
        "num": "100",
    }
    return client.search(search_params).as_dict()


def main():
    scholar_profiles = load_json_file(BASE_DATA_PATH / "scholar_profiles.json")
    author_ids = [profile["author_id"] for profile in scholar_profiles]

    # author_ids = ["CUSTOM_AUTHOR_ID"]

    client = get_serpapi_client()
    save_path = (
        BASE_DATA_PATH / "scholar_crawls" / f"details_{get_current_time_str()}.json"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    result = {}
    for author_id in tqdm.tqdm(author_ids):
        profile_details = fetch_profile_details(client, author_id)
        result[author_id] = profile_details

        save_json_to_file(save_path, result)


if __name__ == "__main__":
    main()
