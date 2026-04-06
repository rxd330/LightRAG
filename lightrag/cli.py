import argparse
import asyncio
import json
import os
import sys

# Try to use aiohttp since it's a core dependency in pyproject.toml
try:
    import aiohttp
except ImportError:
    print("Error: aiohttp is required for the CLI. Please install it using 'pip install aiohttp'.")
    sys.exit(1)


async def main_async():
    parser = argparse.ArgumentParser(
        description="LightRAG CLI - interact with LightRAG API from the terminal"
    )

    # Main command structure
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the LightRAG knowledge base")
    query_parser.add_argument("query", type=str, help="The question or prompt to process")
    query_parser.add_argument(
        "--mode",
        type=str,
        default="mix",
        choices=["local", "global", "hybrid", "naive", "mix", "bypass"],
        help="Query strategy (default: mix)",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top items to retrieve (default: server dependent)",
    )
    query_parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming output and wait for the complete response",
    )
    query_parser.add_argument(
        "--only-need-context",
        action="store_true",
        help="Only return retrieved chunks without an LLM-generated answer",
    )

    # Common connection settings
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("LIGHTRAG_API_URL", "http://127.0.0.1:9621"),
        help="LightRAG API URL (default: http://127.0.0.1:9621 or LIGHTRAG_API_URL env var)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("LIGHTRAG_API_KEY", ""),
        help="LightRAG API Key (default: LIGHTRAG_API_KEY env var)",
    )

    args = parser.parse_args()

    # Headers setup
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    if args.command == "query":
        payload = {
            "query": args.query,
            "mode": args.mode,
        }
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        if args.only_need_context:
            payload["only_need_context"] = True

        # Endpoint selection based on streaming
        endpoint = f"{args.url.rstrip('/')}/query/stream" if args.stream else f"{args.url.rstrip('/')}/query"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Error {response.status}: {error_text}", file=sys.stderr)
                        sys.exit(1)

                    if args.stream:
                        # Streaming mode: Parse NDJSON format
                        first_chunk = True
                        async for line in response.content:
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                if "references" in data and first_chunk:
                                    print("\n[References]", file=sys.stderr)
                                    for idx, ref in enumerate(data["references"], 1):
                                        print(f"  {idx}. {ref.get('file_path', 'Unknown file')}", file=sys.stderr)
                                    print("-" * 50, file=sys.stderr)
                                    first_chunk = False
                                
                                if "response" in data:
                                    # Print incrementally to stdout
                                    print(data["response"], end="", flush=True)
                                    
                                if "error" in data:
                                    print(f"\n[Error from server] {data['error']}", file=sys.stderr)
                            except json.JSONDecodeError:
                                print(f"Error decoding chunk: {line.decode('utf-8')}", file=sys.stderr)
                        print()  # Final newline
                    else:
                        # Non-streaming mode
                        data = await response.json()
                        if "references" in data and data["references"]:
                            print("\n[References]", file=sys.stderr)
                            for idx, ref in enumerate(data["references"], 1):
                                print(f"  {idx}. {ref.get('file_path', 'Unknown file')}", file=sys.stderr)
                            print("-" * 50, file=sys.stderr)
                        
                        if "response" in data:
                            print(data["response"])
                        else:
                            print("Success, but no response payload returned.", file=sys.stderr)

        except aiohttp.ClientError as e:
            print(f"Connection error to {args.url}: {e}", file=sys.stderr)
            print("Make sure your lightrag-server is running.", file=sys.stderr)
            sys.exit(1)


def main():
    """Entry point for the CLI."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
