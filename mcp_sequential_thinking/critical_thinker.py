from typing import Optional, Dict, Any
import openai
import os


class CriticalThinker:
    """Generates critical thinking responses to thoughts."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the critical thinker with an optional API key.

        Args:
            api_key: Optional OpenAI API key. If not provided, will look for OPENAI_API_KEY in environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def generate_critical_response(
        self, thought: str, context: Dict[str, Any] = None
    ) -> Optional[str]:
        """Generate a critical thinking response to a thought.

        Args:
            thought: The thought to analyze
            context: Optional context about the thought

        Returns:
            Optional[str]: The critical response, or None if generation failed
        """
        if not self.api_key:
            return None

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)

            # Prepare the prompt
            system_prompt = """You are a critical thinking assistant. Your role is to provide an objective, 
            constructive critique of the thought process. Consider:
            - Potential logical fallacies or cognitive biases
            - Unexamined assumptions
            - Alternative perspectives
            - Missing context or information
            - Potential improvements or refinements
            
            Be concise, specific, and constructive in your response."""

            user_prompt = f"""Analyze the following thought and provide constructive criticism:
            
            {thought}"""

            if context:
                user_prompt += f"\n\nContext: {context}"

            # Make the API call
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # Using a more capable model for better analysis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            return response.choices[0].message.content.strip()

        except Exception:
            return None
