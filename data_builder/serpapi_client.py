import os

import serpapi


def get_serpapi_client() -> serpapi.Client:
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_api_key is None:
        raise ValueError("SERPAPI_API_KEY environment variable not set")

    client = serpapi.Client(api_key=serpapi_api_key)
    return client
