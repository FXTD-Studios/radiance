"""
WAN 2.2's LoRAs come in high_noise / low_noise pairs, like its two experts: the
Loader puts each file on its own expert (model, model_low_noise).
"""
import pytest

from radiance.nodes.generate import loader

HIGH = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
LOW = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"


@pytest.fixture
def on_disk(monkeypatch):
    files = {HIGH, LOW, "style.safetensors"}
    monkeypatch.setattr(loader.folder_paths, "get_full_path",
                        lambda folder, name: f"/loras/{name}" if name in files else None)
    return files


def test_a_high_noise_lora_brings_its_low_noise_file(on_disk):
    assert loader._wan_moe_lora_stacks([(HIGH, 1.0, 1.0)]) == ([(HIGH, 1.0, 1.0)], [(LOW, 1.0, 0)])


def test_either_file_of_a_pair_or_both_give_the_pair_once(on_disk):
    pair = ([(HIGH, 1.0, 1.0)], [(LOW, 1.0, 0)])
    assert loader._wan_moe_lora_stacks([(LOW, 1.0, 1.0)]) == pair
    assert loader._wan_moe_lora_stacks([(HIGH, 1.0, 1.0), (LOW, 1.0, 1.0)]) == pair


def test_a_lora_without_a_companion_stays_on_the_expert_its_name_says(on_disk):
    on_disk.discard(LOW)
    assert loader._wan_moe_lora_stacks([("style.safetensors", 0.8, 0.5), (HIGH, 1.0, 1.0)]) == (
        [("style.safetensors", 0.8, 0.5), (HIGH, 1.0, 1.0)], [])
    on_disk.discard(HIGH)
    on_disk.add(LOW)
    assert loader._wan_moe_lora_stacks([(LOW, 1.0, 1.0)]) == ([], [(LOW, 1.0, 0)])


def test_a_disabled_entry_does_not_hide_its_active_companion(on_disk):
    assert loader._wan_moe_lora_stacks([(HIGH, 0, 0), (LOW, 1.0, 1.0)]) == ([(HIGH, 1.0, 1.0)], [(LOW, 1.0, 0)])


def test_the_noise_tag_is_read_in_the_file_name_not_its_folder(monkeypatch):
    high, low = "low_noise/style_high_noise.safetensors", "low_noise/style_low_noise.safetensors"
    monkeypatch.setattr(loader.folder_paths, "get_full_path", lambda folder, name: f"/loras/{name}")
    assert loader._find_wan_moe_companion(high) == low
    assert loader._wan_moe_lora_stacks([(high, 1.0, 1.0)]) == ([(high, 1.0, 1.0)], [(low, 1.0, 0)])

