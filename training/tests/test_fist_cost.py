"""The optional cost for standing on the melee slot while a loaded gun is held (training reward only)."""

import numpy as np
import pytest

from training import paths  # noqa: F401
from doomnet import spec
from training.rl import reward as R


def vit(slot, bullets, shells=0.0, sel_ammo=None):
    v = np.zeros(spec.NUM_VITALS, np.float32)
    v[0], v[2], v[4], v[5] = 100.0, slot, bullets, shells
    v[3] = bullets if sel_ammo is None else sel_ammo
    return v


def test_charged_only_on_the_fist_with_bullets_held():
    cost = 0.003
    assert R.fist_charge(vit(1, 50), cost) == pytest.approx(cost)       # fist up, bullets in reserve
    assert R.fist_charge(vit(1, 1), cost) == pytest.approx(cost)
    assert R.fist_charge(vit(1, 0), cost) == 0.0                         # out of bullets: the engine put the fist up
    assert R.fist_charge(vit(1, 0, shells=8), cost) == 0.0               # shells alone may belong to no owned gun
    assert R.fist_charge(vit(2, 50), cost) == 0.0                        # pistol selected
    assert R.fist_charge(vit(3, 50, shells=8), cost) == 0.0              # shotgun selected
    assert R.fist_charge(vit(2, 0), cost) == 0.0                         # empty pistol still selected (dry window)
    assert R.fist_charge(vit(1, 50), 0.0) == 0.0                         # switched off by default
    assert R.fist_charge(None, cost) == 0.0
    assert R.RewardCoef().fist_cost == 0.0


def test_tracker_books_it_under_ammo_on_a_real_map_and_only_then():
    """Real engine: one weapon press from the pistol selects the fist (bullets still held)."""
    from doom_env import make_env
    from training.mapmeta import load_meta
    from fixtures import fixture_wad

    wad = fixture_wad("#####\n#P..#\n#####")
    env = make_env(wad, episode_ticks=400)
    noop = np.zeros(9, np.int8)
    nxt = noop.copy(); nxt[8] = 1

    def run(cost):
        obs, _ = env.reset(seed=0)
        tr = R.RewardTracker(R.RewardCoef(fist_cost=cost), 0, 0, False, True)
        tr.reset(env, env.privileged_state(), float(obs["vitals"][4]))
        ammo_terms, slots = [], []
        for t in range(40):
            a = nxt if t == 5 else noop                                   # one press: pistol -> fist
            obs, *_ = env.step(a)
            _, terms = tr.step(env.privileged_state(), a, float(obs["vitals"][4]), obs["vitals"])
            ammo_terms.append(float(terms[R.T["ammo"]])); slots.append(int(round(float(obs["vitals"][2]))))
        return np.array(ammo_terms), np.array(slots), float(obs["vitals"][4])

    terms, slots, bullets = run(0.003)
    assert bullets > 0 and (slots == 2).any() and (slots == 1).any(), (slots, bullets)
    assert np.allclose(terms[slots == 1], -0.003) and np.allclose(terms[slots != 1], 0.0)
    terms0, slots0, _ = run(0.0)
    assert np.allclose(terms0, 0.0) and (slots0 == slots).all()
    env.close()
