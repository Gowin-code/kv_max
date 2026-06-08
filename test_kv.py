import threading
import time
from kv import KV


def test_user_data():
    kv = KV(base_path="test_data")

    kv.set_user("123", "language", "en")

    assert kv.get_user("123", "language") == "en"


def test_invalid_user_field():
    kv = KV(base_path="test_data")

    kv.set_user("123", "password", "secret")

    assert kv.get_user("123", "password") is None


def test_ttl_persistence():
    kv = KV(base_path="test_data")

    kv.set_with_ttl("temp", "value", 60)
    kv.flush_all()

    kv2 = KV(base_path="test_data")

    assert kv2.get("temp") == "value"


def test_multiple_store_persistence():
    kv = KV(base_path="test_data")

    kv.set_context("store_a")
    kv.set("a", "1")

    kv.set_context("store_b")
    kv.set("b", "2")

    kv.flush_all()

    kv2 = KV(base_path="test_data")

    kv2.set_context("store_a")
    assert kv2.get("a") == "1"

    kv2.set_context("store_b")
    assert kv2.get("b") == "2"


def test_flush_all_no_dirty_store():
    kv = KV(base_path="test_data")

    kv.flush_all()

    assert True


def test_overwrite_key():
    kv = KV(base_path="test_data")

    kv.set("k", "v1")
    kv.set("k", "v2")

    assert kv.get("k") == "v2"


def test_delete_missing_key():
    kv = KV(base_path="test_data")

    kv.delete("missing")

    assert kv.get("missing") is None


def test_concurrent_set():
    kv = KV(base_path="test_data")

    def worker(i):
        kv.set(f"k{i}", i)

    threads = [
        threading.Thread(target=worker, args=(i,))
        for i in range(100)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    for i in range(100):
        assert kv.get(f"k{i}") == i


def test_concurrent_overwrite():
    kv = KV(base_path="test_data")

    def worker():
        for _ in range(100):
            kv.set("shared", "value")

    threads = [threading.Thread(target=worker) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert kv.get("shared") == "value"


def test_shopping_list_persistence():
    kv = KV(base_path="test_data")

    kv.add_to_shopping_list(
        "123",
        {
            "item_name": "Rice",
            "sku_fingerprint": "rice1",
            "quantity": 2
        }
    )

    kv.flush_all()

    kv2 = KV(base_path="test_data")

    items = kv2.get_shopping_list("123")

    assert len(items) == 1


def test_shopping_list_quantity_merge():
    kv = KV(base_path="test_data")

    kv.add_to_shopping_list(
        "123",
        {
            "item_name": "Rice",
            "sku_fingerprint": "rice1",
            "quantity": 2
        }
    )

    kv.add_to_shopping_list(
        "123",
        {
            "item_name": "Rice",
            "sku_fingerprint": "rice1",
            "quantity": 3
        }
    )

    items = kv.get_shopping_list("123")

    assert items[0]["quantity"] == 5


def test_remove_shopping_item():
    kv = KV(base_path="test_data")

    kv.add_to_shopping_list(
        "123",
        {
            "item_name": "Rice",
            "sku_fingerprint": "rice1",
            "quantity": 1
        }
    )

    removed = kv.remove_shopping_item("123", 0)

    assert removed is not None
    assert len(kv.get_shopping_list("123")) == 0


def test_metrics_after_operations():
    kv = KV(base_path="test_data")

    kv.set("a", "b")
    kv.get("a")
    kv.delete("a")

    metrics = kv.get_metrics()

    assert isinstance(metrics, dict)
    assert "flush_worker" in metrics


def test_context_switching():
    kv = KV(base_path="test_data")

    for i in range(20):
        kv.set_context(f"store_{i}")
        kv.set("x", str(i))

    for i in range(20):
        kv.set_context(f"store_{i}")
        assert kv.get("x") == str(i)


def test_large_key_batch():
    kv = KV(base_path="test_data")

    for i in range(1000):
        kv.set(f"key{i}", i)

    assert kv.get("key999") == 999