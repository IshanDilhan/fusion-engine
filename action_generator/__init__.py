"""Action Generator — Lightweight multi-task neural policy module.

Takes [Intent, Motion, Direction, Context] from the upstream fusion engine
and perception runners, and predicts Robot Actions (A01-A15) with continuous
motion control signals (velocity, turning rate, comfort distance).

Designed for real-time deployment on Jetson Orin Nano (<2ms inference).
"""
