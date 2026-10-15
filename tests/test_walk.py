"""The graph walk on a saved graph.

tests/fixtures/graph-sdxl-lora.json stands in for the owner's real export
(Load Checkpoint -> LoraLoader -> two CLIPTextEncode, EmptyLatentImage batch
12 -> KSampler -> VAEDecode -> SaveImage, plus the Review node), built from
the class and input names in ComfyUI's nodes.py. The owner's step
replaces it with a graph exported from a real ComfyUI."""

import copy

from comfyui_trainything.walk import MAX_HOPS, walk_graph


def test_walk_from_review_node(graph):
    facts = walk_graph(graph, "12")
    assert facts["walk"] == "complete"
    assert facts["seed"] == 156680208700286
    assert facts["steps"] == 28
    assert facts["cfg"] == 6.5
    assert facts["sampler"] == "dpmpp_2m"
    assert facts["scheduler"] == "karras"
    assert facts["denoise"] == 1.0
    assert facts["prompt_text"].startswith("pixel art sprite of a red-haired witch")
    assert facts["negative_text"].startswith("blurry")
    assert facts["checkpoint"] == "sd_xl_base_1.0.safetensors"
    assert facts["loras"] == [
        {"name": "pixel-art-xl.safetensors", "strength_model": 0.8, "strength_clip": 0.7}
    ]
    assert (facts["batch_size"], facts["width"], facts["height"]) == (12, 1024, 1024)


def test_walk_from_save_image_is_the_same(graph):
    a = walk_graph(graph, "12")
    b = walk_graph(graph, 9)  # an int id works too
    assert a == b


def test_two_loras_in_application_order(graph):
    g = copy.deepcopy(graph)
    g["11"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["10", 0], "lora_name": "detail.safetensors", "strength_model": 0.5},
    }
    g["3"]["inputs"]["model"] = ["11", 0]
    facts = walk_graph(g, "12")
    assert [l["name"] for l in facts["loras"]] == ["pixel-art-xl.safetensors", "detail.safetensors"]
    assert facts["walk"] == "complete"


def test_unknown_class_stops_and_says_where(graph):
    g = copy.deepcopy(graph)
    g["3"]["class_type"] = "SamplerCustomAdvanced"
    facts = walk_graph(g, "12")
    assert facts["walk"].startswith("stopped:")
    assert "SamplerCustomAdvanced" in facts["walk"] and "node 3" in facts["walk"]
    assert "seed" not in facts


def test_partial_walk_keeps_what_it_found(graph):
    g = copy.deepcopy(graph)
    g["5"]["class_type"] = "LatentUpscale"
    facts = walk_graph(g, "12")
    assert facts["seed"] == 156680208700286 and facts["checkpoint"] == "sd_xl_base_1.0.safetensors"
    assert "batch_size" not in facts
    assert "LatentUpscale" in facts["walk"]


def test_primitive_seed_is_read(graph):
    g = copy.deepcopy(graph)
    g["30"] = {"class_type": "PrimitiveInt", "inputs": {"value": 42}}
    g["3"]["inputs"]["seed"] = ["30", 0]
    assert walk_graph(g, "12")["seed"] == 42


def test_hop_cap_on_a_cycle(graph):
    g = copy.deepcopy(graph)
    g["40"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["41", 0], "scale_by": 1}}
    g["41"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["40", 0], "scale_by": 1}}
    g["12"]["inputs"]["images"] = ["40", 0]
    facts = walk_graph(g, "12")
    assert f"hop limit ({MAX_HOPS})" in facts["walk"]


def test_no_graph():
    assert walk_graph(None, "1")["walk"].startswith("no graph")
    assert walk_graph({"1": {"class_type": "X", "inputs": {}}}, "2")["walk"] == "node 2 is not in the graph"


OWNER_GRAPH = __import__("pathlib").Path(__file__).resolve().parent / "fixtures" / "graph-owner.json"


@__import__("pytest").mark.skipif(not OWNER_GRAPH.exists(), reason="no tests/fixtures/graph-owner.json yet (the owner's exported graph)")
def test_walk_on_the_owners_graph():
    """The owner's step: the owner's real graph, exported as API JSON.
    From every TrainyThing Review / Send in it, the walk reaches the sampler,
    the prompts and the checkpoint."""
    import json

    graph = json.loads(OWNER_GRAPH.read_text())
    starts = [nid for nid, n in graph.items() if n.get("class_type") in ("TrainyThingReview", "TrainyThingSend")]
    assert starts, "graph-owner.json has no TrainyThing Review or Send node"
    for nid in starts:
        facts = walk_graph(graph, nid)
        assert facts["walk"] == "complete", (nid, facts["walk"])
        for key in ("seed", "steps", "cfg", "sampler", "prompt_text", "checkpoint", "batch_size"):
            assert key in facts, (nid, key, facts)
