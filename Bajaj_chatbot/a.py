import json, os, pickle, faiss, numpy as np, ollama
from sentence_transformers import SentenceTransformer
# Model used is "phi"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_data():
    if os.path.exists("data.json"):
        with open("data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def make_chunks(data):
    chunks = []
    # FAQs
    for faq in data.get("faqs", []):
        chunks.append(f"Category: {faq.get('category','')} | Q: {faq.get('question','')} | A: {faq.get('answer','')}")
    # Bikes
    for cat, models in data.get("bikes", {}).items():
        for m in models:
            colors = ", ".join(m.get('available_colors', []) or ['N/A'])
            chunks.append(
                f"Category: {cat} | Model: {m.get('model','')} | Engine: {m.get('engine_cc','')}cc | Power: {m.get('power_ps','')}PS | "
                f"Torque: {m.get('torque_nm','')}Nm | Starting Price: ₹{m.get('starting_price_inr','')} | On-Road Price (Hyderabad): ₹{m.get('on_road_price_hyderabad_inr','N/A')} | "
                f"Colors: {colors} | Features: {m.get('key_features','')}"
            )
    # Customer service
    cs = data.get("customer_service", {}).get("contact_info", {})
    chunks.append(f"Customer Service: Toll-free {cs.get('toll_free_number','')}, Email {cs.get('email','')}, WhatsApp {cs.get('whatsapp_number','')}")
    return chunks

def setup_index(data):
    chunks = make_chunks(data)
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    if os.path.exists("bajaj_index.faiss") and os.path.exists("bajaj_meta.pkl"):
        index = faiss.read_index("bajaj_index.faiss")
        with open("bajaj_meta.pkl", "rb") as f:
            chunks = pickle.load(f)
    else:
        print("Creating FAISS index...")
        embeds = model.encode(chunks).astype(np.float32)
        index = faiss.IndexFlatL2(embeds.shape[1])
        index.add(embeds)
        faiss.write_index(index, "bajaj_index.faiss")
        with open("bajaj_meta.pkl", "wb") as f:
            pickle.dump(chunks, f)
    return index, chunks, model

def get_response(prompt, history=[]):
    # Add recent history
    hist_str = "".join(f"Previous User: {u}\nPrevious Assistant: {b}\n" for u,b in history[-1:])
    # Retrieve context
    query_emb = model.encode([prompt]).astype(np.float32)
    _, idx = index.search(query_emb, k=3)
    context = "\n".join([docs[i] for i in idx[0]])
    prompt = f"""
You are Bajaj Auto's customer care assistant.
Answer the CURRENT QUERY using dataset context.
=== History ===
{hist_str}
=== Dataset Context ===
{context}
=== Current Query ===
{prompt}
=== Answer ===
"""
    try:
        stream = ollama.generate(
            model="phi", 
            prompt=prompt, 
            options={"temperature":0.4, "num_ctx":1000},
            stream=True
        )
        return stream
    except Exception as e:
        return f"Error: Could not connect to Ollama ({e}). Ensure Ollama is running and model is pulled."

if __name__ == "__main__":
    try:
        data = load_data()
        index, docs, model = setup_index(data)
    except Exception as e:
        print(f"Setup error: {e}")
        exit(1)
    print("Hi! Welcome to Bajaj Auto Customer Care. Type 'exit' to quit.\n")
    history = []
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        response = get_response(prompt, history)
        if isinstance(response, str):
            print(f"Bajaj Auto Customer Care: {response}\n")
            history.append((prompt, response))
        else:
            print("Bajaj Auto Customer Care: ", end="", flush=True)  
            full_response = ""
            for chunk in response:
                if 'response' in chunk:
                    print(chunk['response'], end="", flush=True)
                    full_response += chunk['response']
            print("\n")  
            history.append((prompt, full_response))