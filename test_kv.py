from kv import KV
import time


def test_set_get():
    kv = KV(base_path="test_data")
    kv.set("foo", "bar")
    assert kv.get("foo") == "bar"


def test_delete():
    kv = KV(base_path="test_data")
    kv.set("x", "y")
    kv.delete("x")
    assert kv.get("x") is None


def test_ttl():
    kv = KV(base_path="test_data")
    kv.set_with_ttl("a", "b", 1)
    time.sleep(2)
    assert kv.get("a") is None


def test_store_context():
    kv = KV(base_path="test_data")

    kv.set_context("store_a")
    kv.set("x", "1")

    kv.set_context("store_b")
    kv.set("x", "2")

    assert kv.get("x") == "2"

    kv.set_context("store_a")
    assert kv.get("x") == "1"


def test_persistence():
    import os

    kv = KV(base_path="test_data")

    kv.set("persist", "value")
    kv.flush_all()

    print("FILES:", os.listdir("test_data"))

    with open("test_data/global.json", "r") as f:
        print("GLOBAL.JSON CONTENT:")
        print(f.read())

    kv2 = KV(base_path="test_data")

    print("LOADED VALUE:", kv2.get("persist"))

    assert kv2.get("persist") == "value"


def test_shopping_list():
    kv = KV(base_path="test_data")

    kv.add_to_shopping_list(
        "123",
        {
            "item_name": "Rice",
            "sku_fingerprint": "rice1",
            "quantity": 2
        }
    )

    items = kv.get_shopping_list("123")

    assert len(items) == 1


def test_metrics():
    kv = KV(base_path="test_data")

    kv.set("a", "b")
    kv.get("a")

    metrics = kv.get_metrics()

    assert isinstance(metrics, dict)