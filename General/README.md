# General

Shared physics and geometry utilities used across the FDFD, ray-tracing, and profile-generation scripts.

## Code

- `Schwarzchild.py`: defines the Schwarzschild refractive-index model used by the optical black-hole simulations.
- `Kerr_Newman.py`: defines the Kerr-Newman refractive-index model and related profile functions.
- `geodesics.py`: computes reference geodesic trajectories for comparison with optical simulations.
- `discontinuity_radius.py`: estimates discontinuity or transition radii in the optical black-hole profiles.
- `annuli.py`: builds annular discretizations and samples piecewise-constant refractive-index values.
- `cases.py`: stores the standard Schwarzschild and Kerr-Newman parameter sets used for figures.
- `constants.py`: collects shared numerical constants used by plotting and simulation scripts.
- `Profiles/`: scripts for plotting refractive-index profiles and annular approximations.
