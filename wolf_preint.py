#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pinocchio as pin

gravity = np.array((0,0,-9.81))


def compute_current_delta_IMU(b_ab, b_wb, dt):
    return [
        0.5 * b_ab * dt**2,
        b_ab * dt,
        pin.exp3(b_wb * dt)
    ]

def compute_current_delta_FT(fl1, fl2, taul1, taul2, b_pbc, b_wb, pbl1, bRl1, pbl2, bRl2):
    return np.array([
        # 0.5 * b_ab * dt**2,
        # b_ab * dt,
        # pin.exp3(b_wb * dt)
    ])

def compose_delta_IMU(Delta1, Delta2, dt):
    return [
        Delta1[0] + Delta1[1]*dt + Delta1[2] @ Delta2[0],
        Delta1[1] + Delta1[2] @ Delta2[1],
        Delta1[2] @ Delta2[2]
    ]

def compose_delta_FT(Delta1, Delta2):
    pass

def state_plus_delta_IMU(x, Delta, Deltat):
    return [
        x[0] + x[1]*Deltat + 0.5*gravity*Deltat**2 + x[2] @ Delta[0],
        x[1] + gravity*Deltat + x[2] @ Delta[1],
        x[2] @ Delta[2]
    ]