from src.position_sizing import optimize_sizes_cvar


def test_optimize_sizes_cvar_basic():
    # small synthetic returns matrix: 4 scenarios x 3 assets
    returns = [
        [0.01, 0.02, -0.01],
        [0.00, -0.01, 0.005],
        [-0.02, 0.01, 0.0],
        [0.005, 0.005, 0.005],
    ]
    weights = optimize_sizes_cvar(returns, target_cvar=0.05, budget=1.0)
    assert weights is not None
    assert isinstance(weights, list)
    assert len(weights) == 3
    assert sum(weights) <= 1.0 + 1e-6
