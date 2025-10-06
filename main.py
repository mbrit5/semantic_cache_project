import os, time, math, numpy as np
from dotenv import load_dotenv
from google import genai 

load_dotenv()

# client
client = genai.Client()

CACHE_THRESHOLD = 0.75  
CACHE_MODEL = "text-embedding-004"
LLM_MODEL = "models/gemini-2.5-flash"

class SimpleCache:
   #cache, embedding and response
    def __init__(self):
        # store cache
        self.cache = []
        # stores session_id
        self.history = {}
        self.metrics = {"hits": 0, "misses": 0, "total_calls": 0, "llm_calls": 0, "avg_latency_ms": 0.0}

        self.MAX_CACHE_SIZE = 3 

    def _evict_oldest(self):
        if len(self.cache) >= self.MAX_CACHE_SIZE:
            oldest_time = float('inf')
            oldest_index = -1
            for i, entry in enumerate(self.cache):
                if entry.get("last_accessed", 0) < oldest_time:
                    oldest_time = entry["last_accessed"]
                    oldest_index = i
            if oldest_index != -1:
                removed_key = self.cache.pop(oldest_index)["query"]
                print(f"[Cache Eviction] Size limit hit. Removed oldest entry.")
  

    def _get_embedding(self, text):
        # vectors
        response = client.models.embed_content(
            model=CACHE_MODEL, 
            contents=text
        )
        return np.array(response.embeddings[0].values)

    def _get_context_key(self, session_id: str, current_query: str) -> str:
       
        # concat previous turns w/ current query as ONE string
        
        past_turns = self.history.get(session_id, [])
        
        # context string from history
        context_parts = []
        for t in past_turns:
            context_parts.append("User: " + t['query'])
            context_parts.append("AI: " + t['response'])
        context = " ".join(context_parts)
        # context + the new query
        return "CONTEXT " + context + " QUERY " + current_query

    def _cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _generate_llm_response(self, query: str) -> str:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[{"role": "user", "parts": [{"text": query}]}])
        self.metrics["llm_calls"] += 1
        return response.text.strip()


    def ask(self, session_id: str, query: str):
        start = time.time()
        self.metrics["total_calls"] += 1
        # context embedding key
        context_key_text = self._get_context_key(session_id, query)
        emb = self._get_embedding(context_key_text)

        # search cache
        for entry in self.cache:
            sim = self._cosine_similarity(emb, entry["embedding"])
            print(f"[Similarity Check] Score: {sim:.4f} (Threshold: {CACHE_THRESHOLD})")

            if sim >= CACHE_THRESHOLD:
                print(f"[Cache Hit] {sim:.2f}")
                self.metrics["hits"] += 1
                latency = (time.time() - start) * 1000
                
                response = entry["response"]
                cache_hit = True
                break
        else:
            # cache = miss --> call LLM
            print("[Cache Miss] Calling LLM…")
            response = self._generate_llm_response(query)
            self.cache.append({"query": context_key_text, "embedding": emb, "response": response})
            self.history.setdefault(session_id, []).append({"query": query, "response": response})

            self.metrics["misses"] += 1
            cache_hit = False

        latency = (time.time() - start) * 1000
        current_total_latency = self.metrics["avg_latency_ms"] * (self.metrics["total_calls"] - 1)
        self.metrics["avg_latency_ms"] = (current_total_latency + latency) / self.metrics["total_calls"]

        return {"response": response, "cache_hit": cache_hit, "latency_ms": latency}



    def print_metrics(self):
        hit_rate = self.metrics["hits"] / self.metrics["total_calls"] * 100
        print("\n  Cache Metrics")
        print(f"Total Calls: {self.metrics['total_calls']}")
        print(f"Cache Hits: {self.metrics['hits']}")
        print(f"Cache Misses: {self.metrics['misses']}")
        print(f"LLM Calls Made: {self.metrics['llm_calls']}")
        print(f"Cache Hit Rate: {hit_rate:.2f}%")
        print(f"Average Latency: {self.metrics['avg_latency_ms']:.2f} ms")


# test
if __name__ == "__main__":

    cache_system = SimpleCache()

    # two session ids
    session_id_A = "user1" 
    session_id_B = "user2"

    print("First Test")
    
    q1_a = "What is the impact of climate change on corn yields?"
    print(cache_system.ask(session_id_A, q1_a))
    
    q2_a = "How does global warming affect maize productivity?"
    print(cache_system.ask(session_id_A, q2_a)) 

    print("\n Second Test")

    q3_a = "What were the average yields in Iowa last year?"
    print(cache_system.ask(session_id_A, q3_a))
        
    q1_b = "What is the primary export from the American Midwest?"
    print(cache_system.ask(session_id_B, q1_b))

    cache_system.print_metrics()