def project(y: list[float]) -> list[float]:
    """
    Projects a point y onto the simplex x.T * 1 = 1
    minimising (x - y)^2
    Implementation based on https://arxiv.org/abs/1309.1541
    """
    u = sorted(y, reverse=True)
    rho = None

    prefix_sum = 0
    rho_prefix_sum = 0
    for i in range(len(u)):
        prefix_sum += u[i]
        if u[i] + 1 / (i + 1) * (1 - prefix_sum) > 0:
            rho = i + 1
            rho_prefix_sum = prefix_sum

    assert rho is not None, "Unable to project point onto a simplex."

    l = 1 / rho * (1 - rho_prefix_sum)

    y_proj = [max(yi + l, 0.0) for yi in y]

    assert abs(sum(y_proj)- 1) < 1e-5, f"""Projection failed, expected sum: 1. Found: {sum(y_proj)}
translation value: {l}
rho: {rho}
initial list: {y}
projected list: {y_proj}
"""

    return y_proj