"""Graders for scoring agent performance on each task."""

from .base_grader import grade_episode
from .task1_grader import Task1Grader
from .task2_grader import Task2Grader
from .task3_grader import Task3Grader

# task_3_unseen reuses Task3Grader: it has the same template universe,
# verdict/network scoring, and grading criteria as task_3. The only thing
# that differs is the TaskConfig budget regime (queue_size, action_budget,
# n_fraud_rings), which is consumed by the data generator, not the grader.
# Registering it here lets eval_suite drive task_3_unseen episodes through
# the same grade_episode() lookup with no special-casing downstream.
_task3_grader = Task3Grader()
GRADERS = {
    "task_1": Task1Grader(),
    "task_2": Task2Grader(),
    "task_3": _task3_grader,
    "task_3_unseen": _task3_grader,
    "task_4": _task3_grader,
}

__all__ = ["grade_episode", "Task1Grader", "Task2Grader", "Task3Grader", "GRADERS"]
