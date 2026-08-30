"""
Tool registry for function calling. Add new tools by:
1. writing a function with signature (args: dict) -> dict
2. registering it in TOOLS with a name + description + arg schema
3. adding it to TOOL_DESCRIPTIONS_FOR_PROMPT (used in the system prompt)
"""
from app import rag


def _retrieve_documents(args: dict) -> dict:
    query = args.get("query", "")
    if not query:
        return {"error": "query is required"}
    results = rag.retrieve(query)
    if not results:
        return {"results": [], "note": "No documents found in the knowledge base for this query."}
    return {"results": results}


def _calculator(args: dict) -> dict:
    """Small example tool showing tool-calling isn't only for RAG."""
    expression = args.get("expression", "")
    try:
        # restricted eval: only digits/operators, no builtins
        allowed = set("0123456789+-*/(). ")
        if not expression or not set(expression) <= allowed:
            return {"error": "invalid or unsafe expression"}
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


TOOLS = {
    "retrieve_documents": _retrieve_documents,
    "calculator": _calculator,
}

TOOL_DESCRIPTIONS_FOR_PROMPT = """Available tools:
1. retrieve_documents(query: string) -> relevant chunks from the ingested knowledge base.
2. calculator(expression: string) -> evaluates a basic arithmetic expression.
"""


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    fn = TOOLS.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool '{tool_name}'"}
    return fn(tool_args or {})
