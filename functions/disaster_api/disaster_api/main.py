import json
from model import predict

def handler(context, data, files=None):
    try:
        if isinstance(data, str) and data.strip():
            payload = json.loads(data)
        else:
            payload = data or {}

        if not isinstance(payload, dict):
            return json.dumps({"status": "error", "message": "Request body must be a JSON object"})

        result = predict(payload)

        return json.dumps({
            "status": "success",
            "data": result
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })
