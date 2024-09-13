from tqdm import tqdm
from chat_bots import get_chat_model

import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obtain LLM outputs')
    parser.add_argument('--api_key', required=True, type=str)
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_type', type=str, default='OpenAI', choices=['OpenAI', 'Google', 'Groq'])
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)
    if args.resume and os.path.exists(args.output_file):
        with open(args.output_file) as f:
            ps = json.load(f)
    else:
        ps = {}
    bot = get_chat_model(args.model_type, args.model_name, args.api_key)

    changed = False
    for vid in tqdm(data.keys()):
        duration = data[vid]['duration']
        if vid not in ps:
            ps[vid] = data[vid]
        if 'response' not in ps[vid]:
            ps[vid]['response'] = [{} for _ in range(len(data[vid]['sentences']))]
        
        for idx in range(len(data[vid]['sentences'])):
            if len(ps[vid]['response'][idx]) > 0:
                continue
            succ = False
            try:
                ans_json, raw = bot.ask(data[vid]['sentences'][idx])
                succ = True
            except Exception as exp:
                print(exp)
                if isinstance(exp, KeyboardInterrupt):
                    exit()
            
            if succ:
                ps[vid]['response'][idx] = ans_json
                changed = True
        
        if not changed:
            continue

        with open(args.output_file, 'w') as f:
            json.dump(ps, f)
        changed = False

    with open(args.output_file, 'w') as f:
        json.dump(ps, f)
