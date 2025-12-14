import requests
import json
import time

def test_llm_priority():
    print("Testing LLM General Understanding...")
    
    # Message unrelated to "payment" keyword but obviously critical
    msg = "The datacenter cooling system has failed! Temperatures rising."
    
    url = "http://127.0.0.1:8000/process"
    try:
        res = requests.post(url, json={"content": msg})
        data = res.json()
        
        print("\n--- Analysis Result ---")
        print(f"Message: {msg}")
        print(f"Intent: {data['analysis']['intent']}")
        print(f"Priority: {data['analysis']['priority']}")
        print(f"Decision: {data['routing']['decision']}")
        print(f"Reason: {data['routing']['explanation']}")
        
        if data['analysis']['priority'] > 0.7 and "CRITICAL" in data['routing']['decision']:
            print("\n✅ SUCCESS: LLM correctly identified urgency without keywords!")
        else:
            print("\n⚠️  Possible Failure: Priority was not high enough.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_llm_priority()
