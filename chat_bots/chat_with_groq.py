from groq import Groq
from .base_bot import ChatBot

class GroqChatBot(ChatBot):
    def __init__(self, api_key, max_retry=3, model_name="llama3-70b-8192"):
        super().__init__(api_key, max_retry)

        self.client = Groq(api_key=self.api_key)
        self.model_name = model_name

    def get_response(self, prompt, query):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )

        response = completion.choices[0].message.content

        return response

