#Each case carries its own `b_inf` so a single change does not break the
#relationship between b_inf and the radius P_* at which the scalar refractive
#index n(P) diverges 

import numpy as np


SCHWARZSCHILD_CASES = [
    {"name": "schwarzschild_b2", "panel": "a", "b_inf": 1},
    {"name": "schwarzschild_b3", "panel": "b", "b_inf": 3.0},
    {"name": "schwarzschild_b4", "panel": "c", "b_inf": 4.0},
    {"name": "schwarzschild_b5", "panel": "d", "b_inf": 5.2},
]


P0 = 6.0


def kn_horizon(a, rho_Q):
    disc = 1.0 - a**2 - rho_Q**2
    if disc < 0:
        return None  # naked singularity (not allowed for the cases we use)
    return 1.0 + np.sqrt(disc)


def kn_n_divergence_radius(a, rho_Q, b_inf):
    
    inv = a / b_inf
    inner = (1.0 - inv) * (1.0 - inv - rho_Q**2)
    if inner < 0:
        return None
    return (1.0 - inv) + np.sqrt(inner)


def kn_auto_P_min(a, rho_Q, b_inf, safety=0.1):
    
    #Pick the innermost simulated radius for a Kerr-Newman case.
    #P_h : outer event horizon,
    #P_* : radius where n(P) diverges
    #P_min = max(P_h, P_*) + safety.
    P_h = kn_horizon(a, rho_Q)
    P_s = kn_n_divergence_radius(a, rho_Q, b_inf)

    candidates = []
    if P_h is not None:
        candidates.append(P_h)
    if P_s is not None:
        candidates.append(P_s)

    if not candidates:
        raise ValueError(
            f"No valid P_min for a={a}, rho_Q={rho_Q}, b_inf={b_inf}"
        )

    return max(candidates) + safety




#change here to evaluate different impact parameters for KN
B_INF_KN = 5

_kn_specs = [
    # panel,     name,                   a,          rho_Q,        ell_sign
    ("a",  "extremal_kerr_corot",        0.4,        0.8,           +1),
    ("b",  "extremal_rn",                0.0,        1.0,           +1),
    ("c",  "kn_corot",                   0.4,        0.8,           +1),
    ("d",  "kn_counter",                -0.4,       0.8,           +1),
]

KERR_NEWMAN_CASES = [
    {
        "panel": panel,
        "name": name,
        "a": a,
        "rho_Q": rho_Q,
        "ell_sign": ell_sign,
        "b_inf": B_INF_KN,
        "P_min": kn_auto_P_min(a, rho_Q, B_INF_KN),
        "P_max": P0,
        "n_annuli": 21,
    }
    for panel, name, a, rho_Q, ell_sign in _kn_specs
]

if __name__ == "__main__":
    # Quick sanity print: shows the radii used to choose P_min.
    print(f"Kerr-Newman cases with b_inf = {B_INF_KN}:")

    for c in KERR_NEWMAN_CASES:
        a_, rho_Q_, b_ = c["a"], c["rho_Q"], c["b_inf"]
        P_h = kn_horizon(a_, rho_Q_)
        P_s = kn_n_divergence_radius(a_, rho_Q_, b_)

        P_h_str = f"{P_h:.6f}" if P_h is not None else "None"
        P_s_str = f"{P_s:.6f}" if P_s is not None else "None"

        print(
            f"{c['panel']:<6} "
            f"{a_:>8.3f} "
            f"{rho_Q_:>8.3f} "
            f"{P_h_str:>10} "
            f"{P_s_str:>10} "
            f"{c['P_min']:>10.6f}"
        )
