def get_chat_model(model_type, model_name, api_key):
    if model_type == 'Google':
        from .chat_with_gemini import GeminiChatBot
        cls = GeminiChatBot
    elif model_type == 'OpenAI':
        from .chat_with_gpt import GPTChatBot
        cls = GPTChatBot
    elif model_type == 'Groq':
        from .chat_with_groq import GroqChatBot
        cls = GroqChatBot
    else:
        raise NotImplementedError
    
    if model_name is not None:
        return cls(api_key=api_key, model_name=model_name)
    else:
        return cls(api_key=api_key)