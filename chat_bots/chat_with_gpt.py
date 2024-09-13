from openai import OpenAI
from .base_bot import ChatBot


class GPTChatBot(ChatBot):
    def __init__(self, api_key, max_retry=3, model_name="gpt-4-turbo"):
        super().__init__(api_key, max_retry)

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def get_response(self, prompt, query):
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )

        response = completion.choices[0].message.content

        return response
