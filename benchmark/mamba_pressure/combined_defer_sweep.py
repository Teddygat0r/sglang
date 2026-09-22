"""Test caps 8, 2 and 1 with nonblocking prefill deferral in every arm."""
from defer_sweep import experiment as defer_experiment
from tail_sweep import main


async def experiment(root):
    await defer_experiment(root, combined=True)


if __name__ == '__main__':
    main(experiment, __file__)
