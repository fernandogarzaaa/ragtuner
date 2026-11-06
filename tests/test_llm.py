import io
import json
import urllib.error

import ragtuner.llm as llm


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_openrouter_fallback_list_and_usage(monkeypatch):
    sent = []

    def fake_urlopen(req, timeout=0):
        sent.append(json.loads(req.data))
        return _Resp(json.dumps({"model": "nvidia/x:free", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0}}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    call = llm.openrouter_caller("google/a:free, nvidia/x:free")
    assert call("s", "u") == "hi"
    assert sent[0]["models"] == ["google/a:free", "nvidia/x:free"] and "model" not in sent[0]
    assert llm.openrouter_caller("claude-opus-5-5") and llm.OPENROUTER_MODELS["claude-opus-5-5"] == "anthropic/claude-opus-5.5"
    assert llm.USAGE["models"]["nvidia/x:free"] >= 1


def test_post_json_retries_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {"Retry-After": "0"}, None)
        return _Resp(b'{"ok": true}')

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm._post_json("http://x", {}, {}) == {"ok": True} and calls["n"] == 3
