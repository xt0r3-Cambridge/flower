"""Projection utility functions."""


def project_on_simplex(y: list[float]) -> list[float]:
    """
    Project a point onto the probability simplex.

    Project a point y onto the simplex x.T * 1 = 1
    minimising (x - y)^2.
    Implementation based on https://arxiv.org/abs/1309.1541
    """
    u = sorted(y, reverse=True)
    rho = None

    prefix_sum = 0.0
    rho_prefix_sum = 0.0
    for i in range(len(u)):
        prefix_sum += u[i]
        if u[i] + 1 / (i + 1) * (1 - prefix_sum) > 0:
            rho = i + 1
            rho_prefix_sum = prefix_sum

    assert rho is not None, "Unable to project point onto a simplex."

    translation_vector = 1 / rho * (1 - rho_prefix_sum)

    y_proj = [max(yi + translation_vector, 0.0) for yi in y]

    threshold = 1e-5

    assert (
        abs(sum(y_proj) - 1) < threshold
    ), f"""Projection failed, expected sum: 1. Found: {sum(y_proj)}
translation value: {translation_vector}
rho: {rho}
initial list: {y}
projected list: {y_proj}
"""

    return y_proj
