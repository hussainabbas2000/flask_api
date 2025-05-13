from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import json
import time

app = Flask(__name__)
CORS(app)

def single_product_prompt(name, qty):
    system_prompt = """
                You are PriceScoutGPT, a specialist in bulk procurement.  
                Your job: find the three lowest total-cost online offers (unit price × quantity + shipping + taxes)
                for the single product described.  

                TOOLS:
                  - web_search_preview → query web search engines

                INSTRUCTIONS:
                1. Always include “bulk” and the exact quantity in your query.
                2. Restrict to B2B sites (Amazon Business, Grainger, McMaster-Carr, Global Industrial)
                   and any relevant government surplus/reseller outlets.
                3. Ask explicitly for total cost (unit price × quantity + shipping + taxes).
                4. Parse out:
                   • URL  
                   • Seller name  
                   • Unit price  
                   • Shipping  
                   • Taxes  
                   • Total cost
                5. Sort by total cost (lowest first) and return only the top 3.
                6. Output exactly valid JSON, in this form:
                   [
                     {
                       "seller": "...",
                       "unit_price": 0.00,
                       "quantity": N,
                       "shipping": 0.00,
                       "taxes": 0.00,
                       "total_cost": 0.00,
                       "url": "..."
                     },
                     … up to 3 items …
                   ]
                If shipping/taxes aren’t stated, use `null`. Maintain a professional tone.
                STRICTLY output *only* the JSON array. No headings, no notes, no extra text—*exactly* a JSON array literal.
                TEMPERATURE=0.0
                TOP_P=1.0
                MAX_TOKENS=512
                """.strip()  # use full version from your function
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
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port = port, debug=True)
