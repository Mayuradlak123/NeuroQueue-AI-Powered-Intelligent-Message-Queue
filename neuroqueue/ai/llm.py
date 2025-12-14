import os
import json
from groq import Groq
from neuroqueue.config.logger import logger
from neuroqueue.config.settings import settings

class LLMService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found. LLM features will be disabled/mocked.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            logger.info(f"LLM initialized with model: {self.model}")

    def analyze_message(self, content: str):
        """
        Uses LLM to analyze message for Priority, Intent, and Risk.
        Returns a dictionary.
        """
        if not self.client:
            # Fallback if no key provided
            return {
                "priority": 0.5,
                "intent": "general",
                "risk": 0.1,
                "reasoning": "LLM disabled (no key)."
            }

        prompt = f"""
        Analyze the following message for a Message Queue system.
        
        Message: "{content}"
        
        Determine:
        1. Priority (0.0 to 1.0): How urgent is this? 1.0 = Critical/Immediate.
        2. Intent: Category (e.g., 'payment_issue', 'server_outage', 'user_feedback', 'security_alert').
        3. Risk (0.0 to 1.0): Likelihood of this indicating a system failure.
        4. Reasoning: Brief explanation.

        Return ONLY a JSON object:
        {{
            "priority": <float>,
            "intent": "<string>",
            "risk": <float>,
            "reasoning": "<string>"
        }}
        """

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI Message Router. Output valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=0.1,
                response_format={"type": "json_object"}, 
            )
            
            result_json = chat_completion.choices[0].message.content
            return json.loads(result_json)
            
        except Exception as e:
            logger.error(f"LLM Analysis failed: {e}")
            return {
                "priority": 0.5,
                "intent": "error_fallback",
                "risk": 0.0,
                "reasoning": f"LLM Error: {str(e)}"
            }

llm_client = LLMService()
