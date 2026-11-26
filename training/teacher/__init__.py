"""Training-time scaffolding. Never imported by the agent in submission/.

Fair navigation teacher: a hand-written explorer that decides from the agent's own memory map,
pose estimate, observation and encoder scalars, and supplies sub-goal, target, action and
value-field labels for the student network.

    core.py     Teacher and TeacherLabel: the decision rules and the door routine
    grid.py     planning grid read from the memory-map crop, shortest-path search
    steer.py    move, strafe and turn keys from dot and cross products
    policy.py   TeacherPolicy (runs the agent's front end and the teacher) and ExitSensor
    debug.py    text views of the map with the route and the target
    gate.py     monster-free suite builder, gate table and failure classification
"""

from .core import Teacher, TeacherLabel
from .policy import ExitSensor, TeacherPolicy

__all__ = ["Teacher", "TeacherLabel", "TeacherPolicy", "ExitSensor"]
