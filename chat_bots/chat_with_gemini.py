import google.generativeai as genai
from .base_bot import ChatBot


class GeminiChatBot(ChatBot):
    def __init__(self, api_key, max_retry=3, model_name="gemini-pro"):
        super().__init__(api_key, max_retry)

        genai.configure(api_key=self.api_key)
        self.chat_model = genai.GenerativeModel(model_name)

    def get_response(self, prompt, query):
        prompt = prompt + f'User Input: "{query}"'
        response = self.chat_model.generate_content(prompt)

        return response.text
    