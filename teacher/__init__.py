"""
teacher — autoregressive EN->VI Transformer used as the knowledge-distillation teacher.

Trained on the v2 train split only (no test leakage). Design and walkthroughs:
docs/plans/teacher-model.md. Entry point: scripts/train_teacher.py.
"""
