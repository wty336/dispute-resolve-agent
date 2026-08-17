import pytest

from dispute_agent.data.splits import build_dataset_manifest


@pytest.fixture
def dataset_manifest():
    return build_dataset_manifest(seed=20260817)


def test_exact_dataset_sizes_and_ood_buckets(dataset_manifest):
    assert dataset_manifest.counts == {
        "sft_train": 1500, "sft_val": 150, "grpo_train": 700,
        "grpo_val": 100, "id_test": 400, "ood_test": 200,
    }
    assert dataset_manifest.ood_counts == {
        "unseen_combination": 80, "language_style": 60, "tool_noise": 60,
    }
    assert dataset_manifest.human_audit.count == 100


def test_fact_instances_do_not_cross_splits(dataset_manifest):
    instances = [set(split.fact_instance_ids) for split in dataset_manifest.splits.values()]
    assert all(not left & right for i, left in enumerate(instances) for right in instances[i + 1:])
