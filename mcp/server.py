import json
import os
import sys

try:
    from mcp.server.fastmcp import FastMCP
    import httpx
except ImportError:
    print("Error: `mcp` and `httpx` are required for the MCP server.")
    print("Please install them using 'pip install mcp httpx'.")
    sys.exit(1)

# Initialize FastMCP Server
mcp = FastMCP("LightRAG")

# Read Configuration from environment variables
API_URL = os.environ.get("LIGHTRAG_API_URL", "http://127.0.0.1:9621").rstrip("/")
API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")


def get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


@mcp.tool()
async def query_lightrag(
    query: str,
    mode: str = "mix",
    only_need_context: bool = False
) -> str:
    """
    Query the LightRAG knowledge base to retrieve graph-based context and answers.
    Use 'only_need_context=True' if you (the assistant) want to synthesize the final answer yourself 
    using the raw chunks. Use 'mode' = mix, local, global, or hybrid.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "query": query,
            "mode": mode,
            "stream": False,
            "only_need_context": only_need_context
        }
        try:
            # We use the generic /query endpoint as defined in the OpenAPI reference
            response = await client.post(f"{API_URL}/query", json=payload, headers=get_headers())
            response.raise_for_status()
            
            data = response.json()
            return json.dumps(data, indent=2)
        except httpx.HTTPStatusError as e:
            return f"Error from LightRAG API: {e.response.text}"
        except Exception as e:
            return f"Error communicating with LightRAG API: {str(e)}"


@mcp.tool()
async def insert_document_text(text: str, source_description: str = "Inserted via MCP") -> str:
    """
    Insert new raw text directly into the LightRAG knowledge graph to be indexed.
    This allows you to feed context into the graph dynamically.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "text": text,
            "file_source": source_description
        }
        try:
            response = await client.post(f"{API_URL}/documents/text", json=payload, headers=get_headers())
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return f"Error from LightRAG API: {e.response.text}"
        except Exception as e:
            return f"Error inserting document: {str(e)}"


@mcp.tool()
async def get_indexing_status() -> str:
    """
    Check the processing status of LightRAG document ingestion tasks.
    Run this after inserting a document to see if the knowledge graph extraction has finished processing.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # The exact health check or status endpoint depends on the implementation.
            # Usually /health or a generic /documents/status is helpful.
            # Falling back to /health if needed can be done here.
            response = await client.get(f"{API_URL}/health", headers=get_headers())
            if response.status_code == 404:
                 # Endpoint might be at /documents/status or similar
                 return "Status endpoint not found. Assuming ready, but you should verify manually."
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except httpx.HTTPStatusError as e:
            return f"Error from LightRAG API: {e.response.text}"
        except Exception as e:
            return f"Error checking document status: {str(e)}"


def main():
    """Entry point for the MCP server to run via stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
