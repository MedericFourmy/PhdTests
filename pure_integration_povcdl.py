#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pinocchio as pin
from example_robot_data import loadTalos
import curves
from multicontact_api import ContactSequence
import matplotlib.pyplot as plt


lxp = 0.1                           # foot length in positive x direction
lxn = 0.1                           # foot length in negative x direction
lyp = 0.065                         # foot length in positive y direction
lyn = 0.065                         # foot length in negative y direction
lz = 0.107                          # foot sole height with respect to ankle joint

contact_Point = np.ones((3,4))
contact_Point[0, :] = [-lxn, -lxn, lxp, lxp]
contact_Point[1, :] = [-lyn, lyp, -lyn, lyp]
contact_Point[2, :] = 4*[-lz]


def f12TOphi(f12,points=contact_Point):
    phi = pin.Force.Zero()
    for i in range(4):
        phii = pin.Force(f12[i*3:i*3+3],np.zeros(3))
        fMi = pin.SE3(np.eye(3),points[:,i])
        phi += fMi.act(phii)
    return phi


examples_dir = '/home/mfourmy/Documents/Phd_LAAS/tests/centroidalkin/data/tsid_gen/'
file_name = 'sinY_nowaist.cs'

cs = ContactSequence()
cs.loadFromBinary(examples_dir + file_name)


q_traj   = cs.concatenateQtrajectories()
dq_traj  = cs.concatenateDQtrajectories()
ddq_traj = cs.concatenateDDQtrajectories()
c_traj = cs.concatenateCtrajectories()
dc_traj = cs.concatenateDCtrajectories()
Lc_traj = cs.concatenateLtrajectories()
contact_frames = cs.getAllEffectorsInContact()
f_traj_lst = [cs.concatenateContactForceTrajectories(l) for l in contact_frames]
print(contact_frames)

min_ts = q_traj.min()
max_ts = q_traj.max()
print('traj dur (s): ', max_ts - min_ts)


dt = 1e-3  # discretization timespan
t_arr   = np.arange(min_ts, max_ts, dt)
q_arr   = np.array([q_traj(t) for t in t_arr])
dq_arr  = np.array([dq_traj(t) for t in t_arr])
ddq_arr = np.array([ddq_traj(t) for t in t_arr])
c_arr = np.array([c_traj(t) for t in t_arr])
dc_arr = np.array([dc_traj(t) for t in t_arr])
Lc_arr = np.array([Lc_traj(t) for t in t_arr])
f_arr_lst = [np.array([f_traj_lst[i](t) for t in t_arr]) for i in range(len(contact_frames))]
l_wrench_lst = [f12TOphi(f12) for f12 in f_arr_lst[0]]
r_wrench_lst = [f12TOphi(f12) for f12 in f_arr_lst[1]]

N = t_arr.shape[0]  # nb of discretized timestamps


robot = loadTalos()
rmodel = robot.model
rdata = robot.data
contact_frame_ids = [rmodel.getFrameId(l) for l in contact_frames]
print(contact_frame_ids)


p_est_lst = [q_arr[0,0:3]]
oRb = pin.Quaternion(q_arr[0,3:7].reshape((4,1))).toRotationMatrix()
oRb_est_lst = [oRb]
v_est_lst = [oRb @ dq_arr[0,0:3]]
c_est_lst = [c_arr[0,:]]
dc_est_lst = [dc_arr[0,:]]
Lc_est_lst = [Lc_arr[0,:]]

p_gtr_lst = [q_arr[0,0:3]]
oRb_gtr_lst = [oRb]
v_gtr_lst = [oRb @ dq_arr[0,0:3]]
c_gtr_lst = [c_arr[0,:]]
dc_gtr_lst = [dc_arr[0,:]]
Lc_gtr_lst = [Lc_arr[0,:]]

gravity = np.array([0,0,-9.81])

robot.com(robot.q0)
mass = rdata.mass[0] 
print('mass talos: ', mass)

for i in range(1,N):
    #############
    # create measurements from ground truth
    #############

    p_gtr_lst.append(q_arr[i,0:3])
    o_q_b = q_arr[i,3:7]
    oRb_gtr_lst.append(pin.Quaternion(o_q_b.reshape((4,1))).toRotationMatrix())
    v_gtr_lst.append(oRb @ dq_arr[i,0:3])
    c_gtr_lst.append(c_arr[i,:])
    dc_gtr_lst.append(dc_arr[i,:])
    Lc_gtr_lst.append(Lc_arr[i,:])

    # IMU
    b_v = dq_arr[i,0:3]
    b_w = dq_arr[i,3:6]
    b_asp = ddq_arr[i,0:3]
    b_acc = b_asp + np.cross(b_w, b_v)

    # FT
    l_F = l_wrench_lst[i]
    r_F = r_wrench_lst[i]
    q_static = q_arr[i,:].copy()
    q_static[:6] = 0
    q_static[6] = 1  
    robot.forwardKinematics(q_static)  
    bTl = robot.framePlacement(q_static, contact_frame_ids[0])
    bTr = robot.framePlacement(q_static, contact_frame_ids[1])
    b_p_bl = bTl.translation
    b_p_br = bTr.translation
    bRl = bTl.rotation
    bRr = bTr.rotation
    b_p_bc = robot.com(q_static)

    #############
    # integrate one step IMU and wrench measurements
    #############
    # IMU
    p_int = p_est_lst[-1] + v_est_lst[-1]*dt +  0.5*oRb_est_lst[-1] @ b_acc*dt**2
    v_int = v_est_lst[-1] + oRb_est_lst[-1] @ b_acc*dt
    oRb_int = oRb_est_lst[-1] @ pin.exp3(b_w*dt)
    
    # FT
    # c_tot_force = oRb_int @ bTl.rotation @ l_F.linear + bTr.rotation @ r_F.linear
    # c_tot_centr_mom = oRb_int @ (
    #     bTl.rotation @ l_F.angular + bTr.rotation @ r_F.angular + 
    #     np.cross(b_p_bl - b_p_bc, bRl @ l_F.linear) + np.cross(b_p_br - b_p_bc, bRr @ r_F.linear)
    # )

    cTl = pin.SE3(oRb_int@bRl, oRb_int@(b_p_bl - b_p_bc))
    cTr = pin.SE3(oRb_int@bRr, oRb_int@(b_p_br - b_p_bc))

    c_tot_wrench = cTl * l_F + cTr * r_F 
    c_tot_force = c_tot_wrench.linear
    c_tot_centr_mom = c_tot_wrench.angular


    c_int = c_est_lst[-1] + dc_est_lst[-1]*dt + 0.5 * (c_tot_force/mass + gravity) * dt**2
    dc_int = dc_est_lst[-1] + (c_tot_force/mass + gravity) * dt
    Lc_int = Lc_est_lst[-1] + (c_tot_centr_mom/mass) * dt


    p_est_lst.append(p_int)
    oRb_est_lst.append(oRb_int)
    v_est_lst.append(v_int)
    c_est_lst.append(c_int)
    dc_est_lst.append(dc_int)
    Lc_est_lst.append(Lc_int)


# store in arrays what can be
p_est_arr = np.array(p_est_lst)
v_est_arr = np.array(v_est_lst)
c_est_arr = np.array(c_est_lst)
dc_est_arr = np.array(dc_est_lst)
Lc_est_arr = np.array(Lc_est_lst)

p_gtr_arr = np.array(p_gtr_lst)
v_gtr_arr = np.array(v_gtr_lst)
c_gtr_arr = np.array(c_gtr_lst)
dc_gtr_arr = np.array(dc_gtr_lst)
Lc_gtr_arr = np.array(Lc_gtr_lst)


# compute errors
p_err_arr = p_est_arr - p_gtr_arr
v_err_arr = v_est_arr - v_gtr_arr
c_err_arr = c_est_arr - c_gtr_arr
dc_err_arr = dc_est_arr - dc_gtr_arr
Lc_err_arr = Lc_est_arr - Lc_gtr_arr
oRb_err_arr = np.array([pin.log3(oRb_gtr.T @ oRb_est) for oRb_est, oRb_gtr in zip(oRb_est_lst, oRb_gtr_lst)])



#######
# PLOTS
#######
err_arr_lst = [p_err_arr, oRb_err_arr, v_err_arr, c_err_arr, dc_err_arr, Lc_err_arr]
fig_titles = ['P error', 'O error', 'V error', 'C error', 'D error', 'Lc error']

for err_arr, fig_title in zip(err_arr_lst, fig_titles):
    plt.figure(fig_title)
    for axis, (axis_name, c) in enumerate(zip('xyz', 'rgb')):
        plt.plot(t_arr, err_arr[:,axis], c, label='err_'+axis_name)
    plt.legend()
    plt.title(fig_title)

plt.show()