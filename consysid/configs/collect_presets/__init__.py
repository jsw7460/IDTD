"""Collect-time preset configs for SysID data collection.

These presets wrap JaxRLWorld training presets (e.g.
``jaxrlworld.rl.configs.presets.go2_flat.base.Go2FlatConfig``) with SysID-
friendly parameter overrides — PD gains, joint Coulomb friction,
foot-ground friction, etc. — so the "real-world" data we collect from
MuJoCo carries known ground-truth values for the parameters we want
to identify.

JaxRLWorld is the RL training framework and intentionally has no SysID
awareness; collect-side overrides therefore live here in ``consysid``
rather than polluting the training preset hierarchy.
"""
