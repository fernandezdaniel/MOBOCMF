import os
import numpy as np
import plotly.io as pio

from contextlib import contextmanager

from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.plot.trace import optimization_trace_single_method
from ax.utils.testing.mock import fast_botorch_optimize_context_manager
from ax.utils.notebook.plotting import render

from dgpmf.models.gp import ExactGPModel
from dgpmf.utils.myrunner import MyRunner

from experiments.synthetic_experiments.branin import BraninExperiment


# Ax uses Plotly to produce interactive plots. These are great for viewing and analysis,
# though they also lead to large file sizes, which is not ideal for files living in GH.
# Changing the default to `png` strips the interactive components to get around this.
pio.renderers.default = "png"

SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_EVALS = 10 if SMOKE_TEST else 30


@contextmanager
def dummy_context_manager():
    yield


if SMOKE_TEST:
    fast_smoke_test = fast_botorch_optimize_context_manager
else:
    fast_smoke_test = dummy_context_manager

NUM_RANDOM_EVALS = 5

if __name__ == "__main__":

    exp = BraninExperiment(runner=MyRunner())

    sobol = Models.SOBOL(exp.search_space)

    for i in range(NUM_RANDOM_EVALS):
        trial = exp.new_trial(generator_run=sobol.gen(1))
        trial.run()
        trial.mark_completed()
        
    with fast_smoke_test():
        for i in range(NUM_EVALS - NUM_RANDOM_EVALS):
            model_bridge = Models.BOTORCH_MODULAR(
                experiment=exp,
                data=exp.fetch_data(),
                surrogate=Surrogate(ExactGPModel),
            )
            trial = exp.new_trial(generator_run=model_bridge.gen(1))
            trial.run()
            trial.mark_completed()

    exp.trials

    exp.fetch_data().df


    # `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
    # optimization runs, so we wrap out best objectives array in another array.
    objective_means = np.array([[trial.objective_mean for trial in exp.trials.values()]])
    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(objective_means, axis=1),
        optimum=BraninExperiment.optimum,
    )
    render(best_objective_plot)