import json
from datetime import datetime

def generate_alert(prediction, risk_score):
    alert = {
        "time": str(datetime.now()),
        "threat": int(prediction),
        "risk_score": int(risk_score)
    }

    with open("../alerts/alerts.json", "a") as f:
        f.write(json.dumps(alert) + "\n")