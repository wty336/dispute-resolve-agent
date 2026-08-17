import pytest

from dispute_agent.data.generator import generate_fact_instances
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


def _latent_signature(instance):
    return (
        instance.observation.item_name,
        instance.observation.order_amount,
        instance.observation.claim_type,
        instance.observation.buyer_requested_amount,
        instance.ground_truth.true_liability,
        instance.ground_truth.true_loss,
        instance.ground_truth.buyer_strategy,
        instance.ground_truth.merchant_strategy,
        instance.ground_truth.risk_level,
    )


def test_independent_split_streams_and_ood_shifts_are_real():
    train_like = generate_fact_instances(seed=20260817, n=20, start_id=0)
    later_split = generate_fact_instances(seed=20260817, n=20, start_id=1500)
    assert {_latent_signature(x) for x in train_like}.isdisjoint(
        {_latent_signature(x) for x in later_split}
    )

    unseen = generate_fact_instances(
        seed=20260817, n=4, start_id=3000, ood_bucket="unseen_combination"
    )
    language = generate_fact_instances(
        seed=20260817, n=4, start_id=3010, ood_bucket="language_style", language_shift=True
    )
    noisy = generate_fact_instances(
        seed=20260817, n=4, start_id=3020, ood_bucket="tool_noise", tool_noise=True
    )
    assert all(x.metadata["unseen_combination"] for x in unseen)
    assert all(
        x.metadata["language_shift"]
        and x.observation.buyer_claim != f"商品出现问题：{x.observation.claim_type}，要求处理。"
        for x in language
    )
    assert all(x.ground_truth.tool_noise_rate >= 0.35 for x in noisy)


def test_generated_labels_have_consistent_compensation_and_public_evidence():
    instances = generate_fact_instances(seed=20260817, n=600)
    by_liability: dict[str, set[str]] = {}
    labels_by_description: dict[str, set[str]] = {}
    for instance in instances:
        liability = instance.ground_truth.true_liability.value
        description = instance.observation.evidence[0].description
        by_liability.setdefault(liability, set()).add(description)
        labels_by_description.setdefault(description, set()).add(liability)
        if instance.ground_truth.true_liability.value in {"buyer", "none"}:
            assert instance.ground_truth.reasonable_compensation_range == (0.0, 0.0)

    assert set(by_liability) == {"merchant", "buyer", "split", "none"}
    assert all(len(descriptions) >= 3 for descriptions in by_liability.values())
    assert any(len(labels) >= 3 for labels in labels_by_description.values())
