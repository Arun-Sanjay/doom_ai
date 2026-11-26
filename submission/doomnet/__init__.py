"""doomnet: inference-time modules of the agent (odometry, spatial memory, encoders, networks).

Everything here must run inside the evaluation sandbox: numpy + torch only, no privileged state,
no path planning code, no network access. Training-only code lives in training/.
"""
