from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import time

app = Flask(__name__)
CORS(app)

def single_product_prompt(name, qty):
    system_prompt = """<... truncated for brevity ...>"""  # use full version from your function
    user_prompt = f"""<... truncated for brevity ...>"""
    return system_prompt, user_prompt

def fetch_bulk_prices(products, api_key, max_retries=1, retry_delay=1.0):
    client = OpenAI(api_key=api_key)
    all_results = {}
    for p in products:
        name, qty = p["name"], p["quantity"]
        for attempt in range(max_retries + 1):
            sys_p, usr_p = single_product_prompt(name, qty)
            try:
                resp = client.responses.create(
                    model="gpt-4o",
                    tools=[{"type": "web_search_preview"}],
                    instructions=sys_p,
                    input=usr_p,
                    temperature=0.0,
                    top_p=1.0,
                    max_output_tokens=512,
                )
                text = resp.output[1].content[0].text.strip()
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed:
                    all_results[name] = parsed
                    break
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(retry_delay)
                retry_delay *= 2
                usr_p = usr_p.replace("Begin now.", "If no price is found... broaden... Begin now.")
            else:
                all_results[name] = []
    return all_results

@app.route("/bulk-pricing", methods=["POST"])
def get_bulk_pricing():
    data = request.json
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "Missing OpenAI API Key"}), 500
    try:
        products = data["products"]
        results = fetch_bulk_prices(products, api_key)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
