import boto3
import json
import os

REGION   = "us-east-1"
MODEL_ID = "amazon.nova-lite-v1:0"


def bedrock_answer(question: str, context: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=REGION)

    prompt = f"""You are an Indian restaurant assistant.
Answer the user using ONLY the context below.
If the context is not enough, say "I don't know."

Context:
{context}

User question: {question}
"""

    # Nova models support the "messages" format + inferenceConfig
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 300,
            "temperature": 0.5,
            "topP": 0.9
        }
    }

    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    data = json.loads(resp["body"].read())

    # Nova response parsing (try a few common schemas)
    # 1) output.message.content[0].text
    try:
        return data["output"]["message"]["content"][0]["text"].strip()
    except Exception:
        pass

    # 2) outputText (some models)
    if isinstance(data, dict) and "outputText" in data:
        return str(data["outputText"]).strip()

    # 3) results[0].outputText (Titan-ish)
    try:
        return data["results"][0]["outputText"].strip()
    except Exception:
        pass

    # If nothing matched, raise a readable error
    raise RuntimeError(
        "Unexpected Bedrock response format. "
        f"Top-level keys: {list(data.keys())}. "
        f"Full response (truncated): {json.dumps(data)[:800]}"
    )
