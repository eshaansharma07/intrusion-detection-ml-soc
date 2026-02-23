import json
from datetime import datetime

def log_event(prediction, risk_score):
    log = {
        "time": str(datetime.now()),
        "prediction": int(prediction),
        "risk_score": int(risk_score)
    }

    with open("../logs/events.json", "a") as f:
        f.write(json.dumps(log) + "\n")