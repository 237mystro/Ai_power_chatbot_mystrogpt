"""
main.py – Mystro GPT (Hybrid) with Live Web Search Fallback and Short-Term Memory

Usage:
1️⃣ Run → 1 to train the model (produces chatbot_model.pth + dimensions.json).
2️⃣ Run → 2 to load the model and chat. Type '/quit' or 'exit' to leave.
"""

import os
import json
import random
from collections import deque

import nltk
import numpy as np
import requests
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bs4 import BeautifulSoup
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from urllib.parse import quote_plus

# ───────────────────────────────────────────────────────────────────────────────
# Download NLP resources
nltk.download('punkt')
nltk.download('wordnet')

# ───────────────────────────────────────────────────────────────────────────────
# Short-term memory for web searches (last 5 queries)
search_memory = deque(maxlen=5)

# ─────────────── API / Utility Functions ──────────────────────────────────────
def get_live_stock_price(ticker="MSFT") -> str:
    try:
        hist = yf.Ticker(ticker).history(period="1d")
        if hist.empty:
            return f"Mystro_gpt: No data found for {ticker}."
        price = hist['Close'].iloc[-1]
        return f"Mystro_gpt: {ticker} is trading at ${price:.2f}."
    except Exception as e:
        return f"Mystro_gpt:Possible network error. Check ypur connection and try again. Stock fetch error — {e}"

def get_weather(city="Buea") -> str:
    api_key = "6853988b7365e74c17453cc3b877a850"
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return f"Mystro_gpt: Weather API error — {r.json().get('message','Unknown')}"
        d = r.json()
        temp = d['main']['temp']
        desc = d['weather'][0]['description']
        return f"Mystro_gpt: {city} is {temp}°C with {desc}."
    except Exception as e:
        return f"Mystro_gpt:Oops!! Possible network error. check your connection and try again"
        # return f"Mystro_gpt: Weather fetch error — {e}"

def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("Mystro_gpt: %A, %B %d, %Y — %I:%M %p")

def duckduckgo_search(query: str) -> str:
    """Perform a DuckDuckGo scrape, return titles + snippets, cache last 5 searches."""
    try:
        q = quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={q}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        result_divs = soup.find_all("div", class_="result__body", limit=5)
        if not result_divs:
            return "Mystro_gpt: No relevant info found online."

        entries = []
        for div in result_divs:
            title_tag = div.find("a", class_="result__a")
            snippet_tag = div.find("a", class_="result__snippet")
            title = title_tag.get_text(strip=True) if title_tag else "🔗 Untitled"
            snippet = (
                snippet_tag.get_text(strip=True)
                if snippet_tag
                else "No summary available."
            )
            entries.append(f"🔹 {title}\n   ✏️ {snippet}")

        # Cache this result
        search_memory.append((query, entries))

        return "Mystro_gpt (web):\n" + "\n\n".join(entries)
    except Exception as e:
        return f"Mystro_gpt: Web search failed — {e}"

#_______recall last search function_______
def recall_last_search() -> str:
    """Recall the most recent cached web search results."""
    if not search_memory:
        return "Mystro_gpt: Let's get started with something."
    query, entries = search_memory[-1]
    header = f"Mystro_gpt: Recall of your last search for “{query}”:\n"
    return header + "\n\n".join(entries)

# ───────────────────────────────────────────────────────────────────────────────
# Neural network for intent classification
class ChatbotModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)

# ───────────────────────────────────────────────────────────────────────────────
class ChatbotAssistant:
    def __init__(self, intents_path: str):
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.responses = {}
        self.model = None
        self.X = None
        self.y = None

    def tokenize_and_lemmatize(self, text: str) -> list:
        lmtzr = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        return [lmtzr.lemmatize(tok.lower()) for tok in tokens]

    def bag_of_words(self, tokens: list) -> list:
        return [1 if w in tokens else 0 for w in self.vocabulary]

    def parse_intents(self):
        with open(self.intents_path, encoding="utf-8") as f:
            data = json.load(f)
        for intent in data["intents"]:
            tag = intent["tag"]
            self.intents.append(tag)
            self.responses[tag] = intent.get("responses", [])
            for pattern in intent.get("patterns", []):
                toks = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(toks)
                self.documents.append((toks, tag))
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        X, y = [], []
        for toks, tag in self.documents:
            X.append(self.bag_of_words(toks))
            y.append(self.intents.index(tag))
        self.X = np.array(X)
        self.y = np.array(y)

    def train(self, lr=0.001, epochs=100, batch_size=8):
        X_t = torch.tensor(self.X, dtype=torch.float32)
        y_t = torch.tensor(self.y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            loss_sum = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            print(f"Epoch {epoch+1}/{epochs} — Loss: {loss_sum/len(loader):.4f}")

    def save(self, model_path: str, dims_path: str):
        torch.save(self.model.state_dict(), model_path)
        with open(dims_path, "w", encoding="utf-8") as f:
            json.dump(
                {"input_size": self.X.shape[1], "output_size": len(self.intents)}, f
            )

    def load(self, model_path: str, dims_path: str):
        with open(dims_path, encoding="utf-8") as f:
            dims = json.load(f)
        self.model = ChatbotModel(dims["input_size"], dims["output_size"])
        self.model.load_state_dict(torch.load(model_path))

    def process_message(self, message: str) -> str:
        lo = message.lower()

        # — Recall feature —
        if "recall" in lo or "previous search" in lo:
            return recall_last_search()

        # — Direct routing for known APIs —
        if "weather" in lo:
            return get_weather()
        if "stock" in lo or "stocks" in lo:
            return get_live_stock_price()
        if "time" in lo or "date" in lo:
            return get_current_time()

        # — Intent classification —
        tokens = self.tokenize_and_lemmatize(message)
        bow = torch.tensor([self.bag_of_words(tokens)], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(bow)
            probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

        # — Fallback keywords trigger web search —
        fallback_keywords = [
            "define", "search", "lookup", "who is", "what is", "explain"
        ]
        if conf.item() < 0.4 or any(kw in lo for kw in fallback_keywords):
            return duckduckgo_search(message)

        # — Canned responses from intents.json —
        tag = self.intents[idx.item()]
        replies = self.responses.get(tag, ["I do not understand yet."])
        return f"Mystro_gpt: {random.choice(replies)}"


# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    assistant = ChatbotAssistant("intents.json")
    assistant.parse_intents()

    print("\nChoose:\n1 - Train Mystro GPT\n2 - Load Mystro GPT and Chat")
    choice = input("Enter your choice: ").strip()

    if choice == "1":
        assistant.prepare_data()
        assistant.train()
        assistant.save("chatbot_model.pth", "dimensions.json")
        print("Training complete. Model saved.")
    elif choice == "2":
        if os.path.exists("chatbot_model.pth") and os.path.exists("dimensions.json"):
            assistant.load("chatbot_model.pth", "dimensions.json")
            print("Mystro GPT ready! Type '/quit' or 'exit' to leave.")
            while True:
                msg = input("You: ").strip()
                if msg.lower() in ["/quit", "exit"]:
                    print("Mystro_gpt: See you next time! 👋")
                    break
                print(assistant.process_message(msg))
        else:
            print("Model files missing. Run option 1 to train first.")
    else:
        print("Invalid choice. Restart and select 1 or 2.")
