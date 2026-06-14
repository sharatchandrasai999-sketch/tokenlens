from tokenlens.cache import SemanticCache, evaluate_threshold

def test_exact_repeat_is_a_hit():
    c = SemanticCache(threshold=0.9)
    c.put("what is the capital of france", "Paris")
    hit, resp, sim = c.get("what is the capital of france")
    assert hit and resp == "Paris"

def test_unrelated_is_a_miss():
    c = SemanticCache(threshold=0.9)
    c.put("what is the capital of france", "Paris")
    hit, _, _ = c.get("explain quantum entanglement in detail")
    assert not hit

def test_stats_track_hits_and_misses():
    c = SemanticCache(threshold=0.9)
    c.put("a", "1"); c.get("a"); c.get("totally different long query here")
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1

def test_eval_reports_false_hit_rate():
    pairs = [
        ("capital of france", "what's the capital of france", True),
        ("capital of austria", "capital of australia", False),
    ]
    r = evaluate_threshold(pairs, threshold=0.95)
    assert "false_hit_rate" in r and "hit_rate" in r

def test_cache_persists_to_disk(tmp_path):
    p = str(tmp_path / "cache.json")
    c1 = SemanticCache(threshold=0.9, store_path=p)
    c1.put("hello world", "hi")
    c2 = SemanticCache(threshold=0.9, store_path=p)   # fresh instance, same file
    hit, resp, _ = c2.get("hello world")
    assert hit and resp == "hi"
