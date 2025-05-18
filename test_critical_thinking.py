import asyncio
import os
from mcp_sequential_thinking.critical_thinker import CriticalThinker

async def test_critical_thinking():
    # Initialize the critical thinker
    thinker = CriticalThinker(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test thought
    thought = """
    We should implement a new feature that allows users to sort their tasks by priority. 
    This will help users focus on the most important tasks first.
    """
    
    # Context for the thought
    context = {
        "thought_number": 1,
        "total_thoughts": 3,
        "stage": "Analysis",
        "tags": ["feature-request", "user-experience"]
    }
    
    # Generate critical response
    response = await thinker.generate_critical_response(thought, context)
    
    # Print results
    print("\n=== Original Thought ===")
    print(thought.strip())
    
    print("\n=== Critical Response ===")
    print(response or "No response generated (check API key and connection)")

if __name__ == "__main__":
    # Make sure to set your OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
    else:
        asyncio.run(test_critical_thinking())
