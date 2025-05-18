import json
import os
import sys
from typing import List, Optional, Dict, Any

from mcp.server.fastmcp import FastMCP, Context

# Use absolute imports when running as a script
try:
    # When installed as a package
    from .models import ThoughtData, ThoughtStage
    from .storage import ThoughtStorage
    from .analysis import ThoughtAnalyzer
    from .critical_thinker import CriticalThinker
    from .logging_conf import configure_logging
except ImportError:
    # When run directly
    from mcp_sequential_thinking.models import ThoughtData, ThoughtStage
    from mcp_sequential_thinking.storage import ThoughtStorage
    from mcp_sequential_thinking.analysis import ThoughtAnalyzer
    from mcp_sequential_thinking.critical_thinker import CriticalThinker
    from mcp_sequential_thinking.logging_conf import configure_logging

logger = configure_logging("sequential-thinking.server")


mcp = FastMCP("sequential-thinking")

# Initialize storage and critical thinker
storage_dir = os.environ.get("MCP_STORAGE_DIR", None)
storage = ThoughtStorage(storage_dir)
critical_thinker = CriticalThinker()


async def _generate_critical_response(
    thought: str, context: Dict[str, Any], thought_data: ThoughtData
) -> Optional[str]:
    """Generate a critical thinking response for a thought.

    Args:
        thought: The thought content
        context: Additional context about the thought
        thought_data: The thought data object

    Returns:
        Optional[str]: The critical response, or None if generation failed
    """
    try:
        # Prepare context for the critical thinker
        analysis_context = {
            "thought_number": thought_data.thought_number,
            "total_thoughts": thought_data.total_thoughts,
            "stage": thought_data.stage.value,
            "tags": thought_data.tags,
            **context,
        }

        # Generate the critical response
        return await critical_thinker.generate_critical_response(
            thought, analysis_context
        )
    except Exception as e:
        logger.error(f"Error in critical response generation: {str(e)}")
        return None


@mcp.tool()
async def process_thought(
    thought: str,
    thought_number: int,
    total_thoughts: int,
    next_thought_needed: bool,
    stage: str,
    tags: Optional[List[str]] = None,
    axioms_used: Optional[List[str]] = None,
    assumptions_challenged: Optional[List[str]] = None,
    ctx: Optional[Context] = None,
    generate_critical_response: bool = True,
) -> dict:
    """Add a sequential thought with its metadata.

    Args:
        thought: The content of the thought
        thought_number: The sequence number of this thought
        total_thoughts: The total expected thoughts in the sequence
        next_thought_needed: Whether more thoughts are needed after this one
        stage: The thinking stage (Problem Definition, Research, Analysis, Synthesis, Conclusion)
        tags: Optional keywords or categories for the thought
        axioms_used: Optional list of principles or axioms used in this thought
        assumptions_challenged: Optional list of assumptions challenged by this thought
        ctx: Optional MCP context object

    Returns:
        dict: Analysis of the processed thought and optional critical response to use in next thought and hints to use in next thought
    """
    try:
        # Log the request
        logger.info(
            f"Processing thought #{thought_number}/{total_thoughts} in stage '{stage}'"
        )

        # Report progress if context is available
        if ctx:
            ctx.report_progress(thought_number - 1, total_thoughts)

        # Convert stage string to enum
        thought_stage = ThoughtStage.from_string(stage)

        # Prepare context for analysis
        context = {
            "tags": tags or [],
            "axioms_used": axioms_used or [],
            "assumptions_challenged": assumptions_challenged or [],
        }

        # Create thought data object with defaults for optional fields
        thought_data = ThoughtData(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_thought_needed=next_thought_needed,
            stage=thought_stage,
            **context,
        )

        # Generate critical response if enabled and OpenAI API key is available
        critical_response = None
        if generate_critical_response and critical_thinker.api_key:
            critical_response = await _generate_critical_response(
                thought, context, thought_data
            )
            thought_data.critical_response = critical_response

        # Validate and store
        thought_data.validate()
        storage.add_thought(thought_data)

        # Get all thoughts for analysis
        all_thoughts = storage.get_all_thoughts()

        # Analyze the thought
        analysis = ThoughtAnalyzer.analyze_thought(thought_data, all_thoughts)

        # Add critical response to the analysis if available
        if critical_response:
            analysis["criticalResponse"] = critical_response
            logger.info(f"Generated critical response for thought #{thought_number}")

        # Log success
        logger.info(f"Successfully processed thought #{thought_number}")

        return analysis
    except json.JSONDecodeError as e:
        # Log JSON parsing error
        logger.error(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {str(e)}", "status": "failed"}
    except Exception as e:
        # Log error
        logger.error(f"Error processing thought: {str(e)}")

        return {"error": str(e), "status": "failed"}


@mcp.tool()
def generate_summary() -> dict:
    """Generate a summary of the entire thinking process.

    Returns:
        dict: Summary of the thinking process
    """
    try:
        logger.info("Generating thinking process summary")

        # Get all thoughts
        all_thoughts = storage.get_all_thoughts()

        # Generate summary
        return ThoughtAnalyzer.generate_summary(all_thoughts)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {str(e)}", "status": "failed"}
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
def clear_history() -> dict:
    """Clear the thought history.

    Returns:
        dict: Status message
    """
    try:
        logger.info("Clearing thought history")
        storage.clear_history()
        return {"status": "success", "message": "Thought history cleared"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {str(e)}", "status": "failed"}
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
def export_session(file_path: str) -> dict:
    """Export the current thinking session to a file.

    Args:
        file_path: Path to save the exported session

    Returns:
        dict: Status message
    """
    try:
        logger.info(f"Exporting session to {file_path}")
        storage.export_session(file_path)
        return {"status": "success", "message": f"Session exported to {file_path}"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {str(e)}", "status": "failed"}
    except Exception as e:
        logger.error(f"Error exporting session: {str(e)}")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
def import_session(file_path: str) -> dict:
    """Import a thinking session from a file.

    Args:
        file_path: Path to the file to import

    Returns:
        dict: Status message
    """
    try:
        logger.info(f"Importing session from {file_path}")
        storage.import_session(file_path)
        return {"status": "success", "message": f"Session imported from {file_path}"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {str(e)}", "status": "failed"}
    except Exception as e:
        logger.error(f"Error importing session: {str(e)}")
        return {"error": str(e), "status": "failed"}


def main():
    """Entry point for the MCP server."""
    logger.info("Starting Sequential Thinking MCP server")

    # Ensure UTF-8 encoding for stdin/stdout
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding != "utf-8":
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", line_buffering=True
        )
    if hasattr(sys.stdin, "buffer") and sys.stdin.encoding != "utf-8":
        import io

        sys.stdin = io.TextIOWrapper(
            sys.stdin.buffer, encoding="utf-8", line_buffering=True
        )

    # Flush stdout to ensure no buffered content remains
    sys.stdout.flush()

    # Run the MCP server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    # When running the script directly, ensure we're in the right directory
    import os
    import sys

    # Add the parent directory to sys.path if needed
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Print debug information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    logger.info(f"Parent directory added to path: {parent_dir}")

    # Run the server
    main()
