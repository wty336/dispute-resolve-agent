from dispute_agent.domain.schemas import DisputeGroundTruth, DisputeObservation


def test_public_observation_has_no_ground_truth_fields():
    forbidden = set(DisputeGroundTruth.model_fields) & set(DisputeObservation.model_fields)
    assert forbidden == {"case_id"}
    schema = DisputeObservation.model_json_schema()
    text = str(schema).lower()
    assert "true_liability" not in text
    assert "should_escalate" not in text
    assert "tool_information_value" not in text
