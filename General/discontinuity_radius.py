import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

rho2 = 0.5
rho_Q = rho2

a = np.linspace(-0.2, 0.2, 600)
b_vals = np.linspace(1, 7, 120)  

cmap = cm.viridis
norm = mcolors.Normalize(vmin=b_vals.min(), vmax=b_vals.max())

fig, ax = plt.subplots(figsize=(8, 6))

for b in b_vals:
    term = 1 - a / b
    inside_sqrt = term * (term - rho2**2)
    inside_sqrt = np.maximum(inside_sqrt, 0)

    P = 1 - a / b + np.sqrt(inside_sqrt)

    ax.plot(a, P, color=cmap(norm(b)), alpha=0.8)


inside_h = 1 - a**2 - rho_Q**2
inside_h = np.maximum(inside_h, 0)

r_plus = 1 + np.sqrt(inside_h)

ax.plot(
    a,
    r_plus,
    'r--',
    linewidth=3,
    label=r'$r_{+}(a)$'
)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$b$', fontsize=20)

# --- labels ---
ax.set_xlabel(r'$\hat{a}$', fontsize=20)
ax.set_ylabel(r'$P_*$', fontsize=20)
ax.set_title(r'Reversed $P_*(\hat{a}, b)$ with Kerr–Newman horizon', fontsize=16)

ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()