import argparse
import mimetypes
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
# model = "gemma-3-27b-it"


def main():
    parser = argparse.ArgumentParser(description="multimodel query rewriting")

    parser.add_argument("--image", help="path to an image")
    parser.add_argument(
        "--query", type=str, help="text query to rewrite based on the image"
    )

    args = parser.parse_args()
    searchable_query(args.image, args.query)


def searchable_query(image, query):
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"
    with open(image, "rb") as f:
        img = f.read()
        system_prompt = """
        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
        """
        parts = [
            system_prompt,
            types.Part.from_bytes(data=img, mime_type=mime),
            query.strip(),
        ]
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=parts
        )
        if response.text:
            print(f"Rewritten query: {response.text.strip()}")
        else:
            print("No response text returned")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
