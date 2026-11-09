from acbpo.reproducibility import stable_seed


def test_stable_seed_is_repeatable_and_namespaced():
    first = stable_seed("s5-bootstrap", 16, "search_e1", "search_e0")
    assert first == stable_seed("s5-bootstrap", 16, "search_e1", "search_e0")
    assert 0 <= first < 2**32
    assert first != stable_seed("v3-bootstrap", 16, "search_e1", "search_e0")
    assert first != stable_seed("s5-bootstrap", 8, "search_e1", "search_e0")
