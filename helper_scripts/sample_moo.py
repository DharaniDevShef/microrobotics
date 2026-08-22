import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

# 1. Setup a multi-objective problem (DTLZ1 with 3 objectives)
problem = get_problem("dtlz1", n_var=7, n_obj=3)

# 2. Generate reference directions (Das-Dennis structured mesh)
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# 3. Configure NSGA-III algorithm
algorithm = NSGA3(
    ref_dirs=ref_dirs,
    sampling=FloatRandomSampling(),                 # Initial population
    crossover=SBX(prob=0.9, eta=30, vtype=float),  # Simulated Binary Crossover
    mutation=PM(prob=1.0 / 7, eta=20),             # Polynomial Mutation
    pop_size=ref_dirs.shape[0]                       # Match population size to ref_dirs
)

# 4. Execute optimization
res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    seed=1,
    verbose=True
)

# 5. Output Pareto-optimal results
print("Found Pareto front solutions shape:", res.F.shape)
print(res.F[:5])  # Display first 5 objective vectors

from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP

# --- 1. 3D Objective Space Scatter Plot ---
plot = Scatter(title="Objective Space (Pareto Front)")
plot.add(res.F, color="red", s=30)
plot.show()

# --- 2. Parallel Coordinate Plot (Great for High-Dimensional Data) ---
plot = PCP(title="Objective Space (Parallel Coordinates)")
plot.add(res.F)
plot.show()