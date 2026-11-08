
import os
from dotenv import load_dotenv
from google import genai
from hindsight_client import Hindsight

load_dotenv()

# Test Gemini
try:
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    result = gemini.models.generate_content(
        model="gemini-flash-latest",
        contents="Reply with exactly: GEMINI CONNECTED"
    )
    print("GEMINI:", result.text.strip())
except Exception as e:
    print("GEMINI FAILED:", str(e)[:300])

# Test Hindsight
try:
    memory = Hindsight(
        base_url=os.environ["HINDSIGHT_BASE_URL"],
        api_key=os.environ["HINDSIGHT_API_KEY"]
    )

    bank = os.environ["HINDSIGHT_BANK_ID"]

    memory.retain(
        bank_id=bank,
        content="MetricChronicle connection test. Revenue means delivered order value minus refunds."
    )

    recalled = memory.recall(
        bank_id=bank,
        query="What is the revenue definition?"
    )

    print("HINDSIGHT RETAIN + RECALL: SUCCESS")

    reflected = memory.reflect(
        bank_id=bank,
        query="Explain the stored revenue definition."
    )

    print("HINDSIGHT REFLECT: SUCCESS")
    print("REFLECTION:", reflected.text[:300])

    memory.close()

except Exception as e:
    print("HINDSIGHT FAILED:", str(e)[:300])
