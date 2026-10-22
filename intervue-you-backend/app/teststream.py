import time
import json
import requests

URL = "http://127.0.0.1:8000/interview/start/stream"
PAYLOAD = {
    "category": "frontend",
    "difficulty": "junior",
    "stack": ["vue", "react"],
    "mode": "text",
    "time_limit_minutes": 10
}


def test_start_stream():
    print(f"Initiating POST request to {URL}...")
    start_time = time.monotonic()
    first_token_time = None
    session_id = None
    full_text = []

    try:
        response = requests.post(URL, json=PAYLOAD, stream=True)
        response.raise_for_status()

        print("Connection established! Listening for SSE events...\n")

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: "):
                raw_json = line_str[6:]
                try:
                    event = json.loads(raw_json)
                    event_type = event.get("type")

                    if event_type == "session_created":
                        session_id = event.get("session_id")
                        print(f"[Event: session_created] Session ID: {session_id}")

                    elif event_type == "token":
                        # Record TTFT on the exact arrival of the first token event
                        if first_token_time is None:
                            first_token_time = time.monotonic() - start_time
                            print(f"\n⚡ [Time To First Token (TTFT)]: {first_token_time:.2f} seconds\n")
                            print("Streaming question: ", end="", flush=True)

                        token = event.get("content", "")
                        full_text.append(token)
                        print(token, end="", flush=True)

                    elif event_type == "done":
                        total_time = time.monotonic() - start_time
                        print("\n\n" + "=" * 55)
                        print(f"✅ Stream Finished!")
                        print(f"Session ID: {session_id}")
                        print(
                            f"TTFT (First Token Latency): {first_token_time:.2f}s"
                            if first_token_time is not None
                            else "No tokens received"
                        )
                        print(f"Total Response Duration:   {total_time:.2f}s")
                        print("=" * 55)

                except json.JSONDecodeError:
                    continue

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Connection failed: {e}")


if __name__ == "__main__":
    test_start_stream()