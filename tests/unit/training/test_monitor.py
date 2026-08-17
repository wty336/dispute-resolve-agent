from dispute_agent.training.monitor import CollapseMonitor


def test_monitor_pauses_after_fifty_bad_steps():
    monitor = CollapseMonitor(window=50, max_zero_variance_ratio=0.30)
    for _ in range(50):
        monitor.observe(group_rewards=[0.4, 0.4, 0.4, 0.4], valid_rollouts=4)
    assert monitor.should_pause is True
    assert monitor.reason == "zero_variance_group_ratio"
