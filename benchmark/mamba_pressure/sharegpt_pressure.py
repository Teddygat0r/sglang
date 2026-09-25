"""Natural-language shared-prefix replay under a matched constrained cache budget."""

from benchmark_utils import main
from sharegpt_multiturn import experiment as multiturn_experiment


async def experiment(root):
    await multiturn_experiment(root, pressure=True)


if __name__ == "__main__":
    main(experiment, __file__)
