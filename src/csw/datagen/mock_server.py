"""Tiny OpenAI-compatible mock (no GPU): answers chat completions with a canned JSON row so the whole pipeline can be tested end to end.  python -m csw.datagen.mock_server [port]"""
import json, sys, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, o, code=200):
        b = json.dumps(o).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"data": [{"id": "mock"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"]; m = re.search(r"ENG: (.*)", user, re.S)
        eng = (m.group(1).strip() if m else "The family discussed the matter and decided to meet again on the following day.")
        row = {"frame": "mock", "eng_initial": "The family talked it through and said see you tomorrow!", "eng": "The family discussed the matter and decided to meet again on the following day." if not m else "MODEL REWROTE THE GIVEN ENGLISH",
               "m_text": "Mock m_text one two three four five six seven eight nine ten eleven twelve.", "hm": "Mock hm one two three four five six seven eight nine ten eleven twelve more.",
               "ood": "Mock ood one two three four five six seven eight nine ten eleven twelve more words here and there to be longer.", "ood_relation": "idiomatic_paraphrase",
               "devices": ["a (b)", "c (d)", "e (f)"], "self_check": ["ok"], "conf": {"m_text": .8, "hm": .8, "ood": .8}, "flags": [], "difficulty": 4}
        txt = json.dumps(row)
        self._send({"id": "x", "model": body.get("model", "mock"), "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": txt}}], "usage": {"prompt_tokens": 9000, "completion_tokens": 400, "total_tokens": 9400}, "_extra_body_seen": body.get("repetition_penalty") or body.get("top_k")})
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
