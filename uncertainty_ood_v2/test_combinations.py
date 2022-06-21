import itertools

# General settings
base_settings = {"epochs": 10}

# Grid search
grid = {"batch_size": [32, 64, 128], "learning_rate": [1e-4, 1e-3, 1e-2]}


# Loop over al grid search combinations
for values in itertools.product(*grid.values()):
    point = dict(zip(grid.keys(), values))
    settings = {**base_settings, **point}
