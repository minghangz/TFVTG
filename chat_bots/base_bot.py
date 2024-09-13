from . import prompts
import json5
import time

def loads(s):
    return json5.loads(s)

class ChatBot:
    def __init__(self, api_key, max_retry=3):
        self.api_key = api_key
        self.max_retry = max_retry
        self.prompt = prompts.v3
    
    def get_response(self, prompt, query):
        pass

    def ask(self, query):
        succ = False
        for _ in range(self.max_retry):
            try:
                response = self.get_response(self.prompt, query)
                response_json = loads(response)
                succ = True
                break
            except Exception as exp:
                print(exp)
                print(response.text)
                time.sleep(1)
        
        if not succ:
            raise exp

        raw = [{"query": query, "response": response_json}]
        
        return response_json, raw
    