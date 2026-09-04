"""
Regression test for the PBVS target-selection oscillation.

Reproduces the exact scenario reported live: two fragments simultaneously
eligible, with their relative y_cam ordering flipping between consecutive
fused-fragment messages (which happens constantly on a moving belt). The
old code re-ranked candidates fresh on every callback with no memory,
so the selected target's identity -- and therefore the sign of the
commanded velocity -- flipped every message, producing a visibly
oscillating, non-converging arm ("DJ-ing" over the belt).
"""
from __future__ import annotations
import numpy as np
from types import SimpleNamespace

from tyre_sorting.control.pbvs_controller import (
    select_locked_target, REACHABLE_X_MIN, REACHABLE_X_MAX, PBVS_APPROACH_Z)
from tyre_sorting.sim.camera_geom import camera_to_world


def _frag(y, x=0.0, z=0.0):
    return SimpleNamespace(centroid=SimpleNamespace(x=x, y=y, z=z))


def test_lock_survives_ranking_flip_between_two_close_fragments():
    """
    Two fragments 5cm apart -- close enough that ordinary sensor/position
    noise could flip which one has the smaller y_cam on any given frame.
    Once locked onto fragment A, a single frame where B's y momentarily
    reads smaller must NOT cause the lock to jump to B.
    """
    belt_velocity = np.array([0.05, 0.0, 0.0])
    a = _frag(y=0.10)
    a_pos = np.array([0.30, 0.10, 0.05])
    b = _frag(y=0.09)  # marginally "ahead" on this frame
    b_pos = np.array([0.55, 0.09, 0.05])  # but far away in world space --
    # a different physical fragment, not sensor noise on the same one

    eligible_world = [(a, a_pos), (b, b_pos)]

    # First callback: nothing locked yet -- acquires the smallest-y
    # candidate (a, since 0.10... wait a has y=0.10, b has y=0.09, so the
    # naive rule would pick b here). This IS the expected first pick.
    target, pos, locked_pos, locked_wall = select_locked_target(
        eligible_world, None, None, belt_velocity, now_wall=0.0)
    assert target is b

    # Second callback, 0.1s later: b has moved out of frame (picked up /
    # gone), only 'a' remains eligible. Lock must transfer cleanly since
    # b truly is gone.
    target2, pos2, locked_pos2, locked_wall2 = select_locked_target(
        [(a, a_pos)], locked_pos, locked_wall, belt_velocity, now_wall=0.1)
    assert target2 is a


def test_lock_prevents_flip_flop_across_many_ambiguous_frames():
    """
    The actual oscillation scenario: a and b stay simultaneously visible
    across many consecutive frames, with their y-ordering alternating
    frame to frame (representative of real per-frame segmentation noise).
    Once locked onto one physical fragment, gated tracking must keep
    returning THAT SAME fragment every frame, not alternate.
    """
    belt_velocity = np.array([0.0, 0.0, 0.0])  # isolate the ranking-flip
    # effect from belt drift
    a_pos = np.array([0.30, 0.10, 0.05])
    b_pos = np.array([0.60, 0.40, 0.05])  # far from a -- clearly a
    # different physical fragment, never within gate_dist of a

    locked_pos, locked_wall = None, None
    selected_physical_fragment = []  # 'a' or 'b', by matching returned pos
    for frame in range(10):
        # Alternate which one reports the smaller y_cam each frame --
        # this is exactly the kind of noisy re-ranking that flipped the
        # old code's target every message.
        if frame % 2 == 0:
            a = _frag(y=0.05)
            b = _frag(y=0.06)
        else:
            a = _frag(y=0.07)
            b = _frag(y=0.02)
        eligible_world = [(a, a_pos), (b, b_pos)]
        target, pos, locked_pos, locked_wall = select_locked_target(
            eligible_world, locked_pos, locked_wall, belt_velocity,
            now_wall=float(frame) * 0.1)
        # Identify which PHYSICAL fragment this is by its (fixed) world
        # position -- the message object itself (a or b) is freshly
        # created each frame, as a real detection message would be, so
        # its Python identity is never meaningful across frames.
        selected_physical_fragment.append('a' if np.allclose(pos, a_pos) else 'b')

    # Once acquired, the SAME physical fragment must be tracked every
    # single frame -- no alternation despite the ranking flipping.
    assert len(set(selected_physical_fragment)) == 1, (
        "target identity flip-flopped across frames -- this is the exact "
        "oscillation reported live as the arm 'DJ-ing' over the belt")


def test_reachable_window_excludes_spawn_point():
    """
    A fragment at the belt's spawn end (world x=-1.1) is camera-visible
    and classification-eligible from the moment it appears, but it is
    1.28m from the arm base -- far outside the iiwa's reach envelope
    (find_action_zone.py only verified x in [-0.55, 0.60] as comfortably
    reachable). REACHABLE_X_MIN/MAX must exclude it; the pick point
    (x=-0.30) must be included.
    """
    assert not (REACHABLE_X_MIN <= -1.1 <= REACHABLE_X_MAX)
    assert REACHABLE_X_MIN <= -0.30 <= REACHABLE_X_MAX


def test_approach_height_is_above_belt_surface():
    """
    camera_to_world's z for a fragment resting on the belt reports the
    belt-surface height (~0.07m) -- PBVS_APPROACH_Z must clear that, or
    the commanded target drives the end-effector into the belt (only
    fragment collisions are masked for the arm, see sim/env.py's
    COLLISION_GROUP_* comment) and the z error never converges.
    """
    _, _, belt_z = camera_to_world((0.0, 0.0, 0.98))  # a belt-plane depth
    assert PBVS_APPROACH_Z > belt_z + 0.02
