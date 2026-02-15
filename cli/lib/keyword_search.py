from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movie


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movie()
    result = []
    for movie in movies:
        if query in movie["title"]:
            result.append(movie)
            if len(result) >= limit:
                break
    return result
