"""Map generator: ASCII grid layouts compiled to UDMF maps. Training-time scaffolding, never imported by the agent.

    udmf.py       grid -> PWAD compiler in the dialect of the reference maps, and the legend
    solver.py     search over (cell, keys held): solvability and route cost
    layouts.py    seeded procedural layouts for tiers 0-4
    randomize.py  domain randomisation of start pose, things, light, textures and door timing
    walker.py     reference walker that follows the solver's route in the engine (tests only)
    cli.py        python -m training.mapgen.cli: training sets, frozen suites, statistics
"""
