from .SUMOEnv import SUMOEnv

# Expose environment to PyMARL2
REGISTRY = {}
REGISTRY["sumo"] = lambda **kwargs: SUMOEnv(**kwargs)
