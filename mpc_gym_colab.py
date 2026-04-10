#!/usr/bin/env python3

import math
import os
import time
import csv
from dataclasses import dataclass, field

import numpy as np
import casadi as ca
from scipy.sparse import block_diag
import matplotlib

matplotlib.use("Agg")                  

import matplotlib.pyplot as plt
import f1tenth_gym                     

# Configurable parameters in the environment
#####################################################################################

@dataclass
class mpc_config:
    NXK: int = 4
    NU:  int = 2
    TK:  int = 10         #2 seconds lookahead at 0.2s timestep

    #User-defined weights for the MPC cost function (tune these!)
    Rk:  list = field(default_factory=lambda: np.diag([1.0, 1.0]))
    Rdk: list = field(default_factory=lambda: np.diag([0.6, 0.6]))
    Qk:  list = field(default_factory=lambda: np.diag([50.0, 50.0, 5.0, 10.0]))
    Qfk: list = field(default_factory=lambda: np.diag([500.0, 500.0, 5.0, 50.0]))
    #############################################################

    N_IND_SEARCH: int   = 20

    DTK:          float = 0.2
    dlk:          float = 0.03
    WB:           float = 0.33

    #Actuation limits
    MIN_STEER:    float = -0.4189
    MAX_STEER:    float =  0.4189
    MAX_DSTEER:   float = np.deg2rad(180.0)
    MAX_SPEED:    float = 5.0
    MIN_SPEED:    float = 0.0
    MAX_ACCEL:    float = 3.0

    # Corridor constraints - "Safety buffer"
    CORRIDOR_FRACTION: float = 0.2


@dataclass
class State:
    x:    float = 0.0
    y:    float = 0.0
    delta: float = 0.0
    v:    float = 0.0
    yaw:  float = 0.0


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def calc_speed_profile(cx, cy, cyaw):
    ncourse = len(cx)

    sp = np.full(ncourse, 5.0) # Pre-initialization of speed at all points to 5.0 m/s
    
    CURVE_THRESHOLD    = 0.01  # Yaw change threshold for detecting curves
    
    CURVE_SPEED        = 2.5   # Reduction in speed at curve points
    ENTRY_BOOST_SPEED  = 5.0
    ENTRY_BOOST_COUNT  = 5
    EXIT_RAMP_COUNT    = 10

    is_curve = np.zeros(ncourse, dtype=bool)
    for i in range(ncourse):
        if abs(cyaw[(i + 1) % ncourse] - cyaw[i]) > CURVE_THRESHOLD:
            is_curve[i] = True                                       # Curve detection based on yaw change
    
    for i in range(ncourse):    
        if is_curve[i]:
            sp[i] = CURVE_SPEED     
            continue
        
        for k in range(1, ENTRY_BOOST_COUNT + 1):
            if is_curve[(i + k) % ncourse]:
                sp[i] = ENTRY_BOOST_SPEED                             #Entry speed is set at 5.0 m/s for ENTRY_BOOST_COUNT points leading into a curve
                break
        
        #Speed drops to CURVE_SPEED, then the speed is linearly interpolated back to 5.0 m/s over EXIT_RAMP_COUNT points after ENTRY_BOOST_COUNT points
        for k in range(1, EXIT_RAMP_COUNT + 1):                       
            if is_curve[(i - k) % ncourse]:
                sp[i] = CURVE_SPEED + (5.0 - CURVE_SPEED) * (k / EXIT_RAMP_COUNT)
                break
    return sp


# Headless MPC  (All ROS stripped)
######################################################################################

class HeadlessMPC:
    """
    Contains every MPC method from mpc_node.py with the ROS calls removed.
    """

    def __init__(self, waypoints_csv: str, border_coeffs_csv: str = None):
        self.config  = mpc_config()
        self.oa      = None
        self.odelta_v = None
        self.odelta   = None
        self._prev_xk = None
        self._prev_uk = None
        self.target_ind = 0
        self.target_ind_initialized = False
        self.prev_odom_yaw = None
        self.yaw_offset    = 0.0
        self._last_ind_list = None

        self.waypoints = self._load_waypoints(waypoints_csv)         # An array of {x, y, yaw} from the waypoints.csv file

        if border_coeffs_csv is not None:            
            self.border_coeffs = self._load_border_coeffs(border_coeffs_csv)  # An array of {w_r, w_l,a, b, tight_cl, tight_cr} is loaded to use as constraints on the border of the race-track.
            self._has_border = True
            print(f" Border constraints loaded: {self.border_coeffs.shape[0]} rows ")
       
        else:
            self.border_coeffs = None
            self._has_border = False

        self._linear_mpc_prob_init()

    ##############################################################################

    def _load_waypoints(self, path: str) -> np.ndarray:
        data = []

        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)                                      # Skip header of the file
            
            for row in reader:
                x, y, yaw = map(float, row)
                data.append([x, y, yaw])
        
        wp = np.array(data)
        wp[:, 2] = np.unwrap(wp[:, 2])
        x0, y0, _ = wp[0]
        x1, y1, _ = wp[-1]
        gap = np.hypot(x1 - x0, y1 - y0)

         # If the gap between the last and first waypoint is large,
         # we interpolate to create a smooth loop.
        if gap > 0.05:
            N_interp = max(int(gap / 0.03), 2)
            t = np.linspace(0.0, 1.0, N_interp + 2)[1:-1]
            bridge = np.vstack([x1 + t*(x0-x1), y1 + t*(y0-y1), np.zeros(len(t))]).T
            wp = np.vstack([wp, bridge])

        diffs = np.diff(wp[:, :2], axis=0)

        s = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:,0], diffs[:,1]))]) #Cumulative distance along the waypoints
        s_new  = np.arange(0.0, s[-1], 0.03)

        # New waypoints interpolated at regular 0.03m intervals along the track + unwrapped yaw angles
        wp_x   = np.interp(s_new, s, wp[:, 0])
        wp_y   = np.interp(s_new, s, wp[:, 1])
        wp_yaw = np.interp(s_new, s, wp[:, 2])

        return np.vstack([wp_x, wp_y, wp_yaw]).T

    ####################################################################################

    def _load_border_coeffs(self, path: str) -> np.ndarray:
        
        data = []

        with open(path, 'r') as f:
            reader = csv.DictReader(f)   #DictReader() is used to read the CSV file as a dictionary; accessing by column names.
            for row in reader:
                data.append([
                    float(row['a']),
                    float(row['b']),
                    float(row['c_center']),
                    float(row['c_left']),
                    float(row['c_right']),
                ])
        
        bc = np.array(data)  
        
        # Since new waypoints are interpolated at regular 0.03m intervals,
        # we use linear interpolation for the border coefficients to match the new waypoints.
        N_orig   = len(bc)
        N_wp     = len(self.waypoints)
        idx_orig = np.arange(N_orig, dtype=float)
        idx_new  = np.linspace(0.0, N_orig - 1, N_wp)

        bc_rs = np.column_stack([
            np.interp(idx_new, idx_orig, bc[:, i]) for i in range(5)
        ])                   
        a_rs, b_rs, cc_rs, cl_rs, cr_rs = bc_rs.T
        # Resampled border coefficients are now aligned with the waypoints

        wp_x = self.waypoints[:, 0]
        wp_y = self.waypoints[:, 1]

        # Track wall XY - Offset raceline by (c_wall - c_center) × normal
        self.border_walls_xy = np.column_stack([
            wp_x + a_rs * (cl_rs - cc_rs),   # Left  wall x
            wp_y + b_rs * (cl_rs - cc_rs),   # Left  wall y
            wp_x + a_rs * (cr_rs - cc_rs),   # Right wall x
            wp_y + b_rs * (cr_rs - cc_rs),   # Right wall y
        ])

        # Tight corridor bounds (CORRIDOR_FRACTION of each half-width) 
        frac     = self.config.CORRIDOR_FRACTION
        tight_cl = cc_rs + frac * (cl_rs - cc_rs)
        tight_cr = cc_rs - frac * (cc_rs - cr_rs)

        self.border_corridor_xy = np.column_stack([
            wp_x + a_rs * (tight_cl - cc_rs),   # Left  corridor x
            wp_y + b_rs * (tight_cl - cc_rs),   # Left  corridor y
            wp_x + a_rs * (tight_cr - cc_rs),   # Right corridor x
            wp_y + b_rs * (tight_cr - cc_rs),   # Right corridor y
        ])

        return np.column_stack([a_rs, b_rs, tight_cl, tight_cr])   

    ##################################################################################

    def _linear_mpc_prob_init(self):
        NX = self.config.NXK
        NU = self.config.NU
        T  = self.config.TK

        self.opti = ca.Opti() # Optimization problem instance

        self.xk = self.opti.variable(NX, T + 1) # State trajectory over the horizon 
        self.uk = self.opti.variable(NU, T)     # Control trajectory over the horizon

        self.x0k        = self.opti.parameter(NX)         # Initial state parameter
        self.ref_traj_k = self.opti.parameter(NX, T + 1)  # Reference trajectory parameter

        ##############################################
        self.Ak_param   = self.opti.parameter(NX * T, NX * T) 
        self.Bk_param   = self.opti.parameter(NX * T, NU * T)
        self.Ck_param   = self.opti.parameter(NX * T)

        R_block  = block_diag([self.config.Rk]  * T).toarray()
        Rd_block = block_diag([self.config.Rdk] * (T - 1)).toarray()
        Q_block  = block_diag([self.config.Qk]  * T + [self.config.Qfk]).toarray()
        ################################################
        # Cost function evaluation
        u_vec = ca.vec(self.uk)              # Control inputs flattened into a vector 
        x_err = ca.vec(self.xk - self.ref_traj_k)  # State error flattened into a vector
        du    = ca.vec(self.uk[:, 1:] - self.uk[:, :-1])  # Control input changes flattened into a vector

        obj   = (ca.mtimes([u_vec.T, R_block, u_vec])
               + ca.mtimes([x_err.T, Q_block, x_err])
               + ca.mtimes([du.T, Rd_block, du]))
        self.opti.minimize(obj)

        # Linearized dynamics constraints
        x_next = ca.vec(self.xk[:, 1:])
        x_curr = ca.vec(self.xk[:, :-1])
        u_flat = ca.vec(self.uk)
        self.opti.subject_to(
            x_next == ca.mtimes(self.Ak_param, x_curr)
                    + ca.mtimes(self.Bk_param, u_flat)
                    + self.Ck_param
        )

        # Actuation and state constraints
        self.opti.subject_to(ca.fabs(self.uk[0, 1:] - self.uk[0, :-1]) <= 5.0)
        self.opti.subject_to(
            ca.fabs(self.uk[1, 1:] - self.uk[1, :-1])
            <= self.config.MAX_DSTEER * self.config.DTK
        )
        self.opti.subject_to(self.xk[:, 0] == self.x0k)
        self.opti.subject_to(self.xk[2, :] >= self.config.MIN_SPEED)
        self.opti.subject_to(self.xk[2, :] <= self.config.MAX_SPEED)
        self.opti.subject_to(ca.fabs(self.uk[0, :]) <= self.config.MAX_ACCEL)
        self.opti.subject_to(ca.fabs(self.uk[1, :]) <= self.config.MAX_STEER)

        # Track-corridor linear constraints
        if self._has_border:
            # Per horizon step
            self.border_a_k  = self.opti.parameter(T + 1)   # normal-x component
            self.border_b_k  = self.opti.parameter(T + 1)   # normal-y component
            self.border_cl_k = self.opti.parameter(T + 1)   # left  limit (outer)
            self.border_cr_k = self.opti.parameter(T + 1)   # right limit (inner)
            
            # Project each point onto the normal vector defined by (a, b)
            # and constrain it to be between the "fractioned" left and right limits.
            proj = (self.border_a_k * ca.vec(self.xk[0, :])
                  + self.border_b_k * ca.vec(self.xk[1, :]))
            self.opti.subject_to(proj <= self.border_cl_k)
            self.opti.subject_to(proj >= self.border_cr_k)

        self.opti.solver('ipopt', {
            'ipopt.print_level': 0, 'print_time': 0,
            'ipopt.max_iter': 200,    
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 5e-2,
            'ipopt.acceptable_iter': 3,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-6,
            'ipopt.warm_start_mult_bound_push': 1e-6,
        })

    #####################################################################
    def _get_model_matrix(self, v, phi, delta):
        # Linearized kinematic bicycle model
        A = np.eye(self.config.NXK)
        A[0, 2] =  self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] =  self.config.DTK * math.sin(phi)
        A[1, 3] =  self.config.DTK * v * math.cos(phi)
        A[3, 2] =  self.config.DTK * math.tan(delta) / self.config.WB

        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] =  self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)
        return A, B, C

    #####################################################################
    def update_state(self, state, a, delta):
        # delta : steering angle in radians
        # a     : acceleration in m/s^2 ; WB : wheelbase in m
        # v/WB * tan(delta) : yaw rate in radians/s
        delta = np.clip(delta, -self.config.MAX_STEER, self.config.MAX_STEER)

        state.x   += state.v * math.cos(state.yaw) * self.config.DTK
        state.y   += state.v * math.sin(state.yaw) * self.config.DTK

        state.yaw += (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        state.v    = np.clip(state.v + a * self.config.DTK,
                             self.config.MIN_SPEED, self.config.MAX_SPEED)
        return state

    #####################################################################
    def _predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0

        for i in range(len(x0)):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], v=x0[2], yaw=x0[3])
        for i in range(1, self.config.TK + 1):
            state = self.update_state(state, oa[i-1], od[i-1])
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw
        return path_predict

    #####################################################################
    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):

        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse  = len(cx)

        if not self.target_ind_initialized:
            dists = (cx - state.x)**2 + (cy - state.y)**2
            self.target_ind = int(np.argmin(dists))
            self.target_ind_initialized = True
        # search_indices : indices of the waypoints to consider for the nearest point search
        # centered around the current target_ind and wrapping around the track if necessary.
        
        # local_best : index of the closest waypoint among the search_indices, to update target_ind for better tracking 
        search_indices = [(self.target_ind + i) % ncourse
                          for i in range(self.config.N_IND_SEARCH)]
        dx  = np.array([cx[i] for i in search_indices]) - state.x
        dy  = np.array([cy[i] for i in search_indices]) - state.y
        local_best = np.argmin(dx**2 + dy**2)


        if local_best > 0:
            self.target_ind = search_indices[local_best]
        ind   = self.target_ind

        # Distance covered per MPC timestep at current speed (floor 0.5 m/s at startup)
        travel = max(state.v, 0.5) * self.config.DTK
        dind   = travel / self.config.dlk
        ind_list = (int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)) % ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        for i, idx in enumerate(ind_list):
            ref_traj[3, i] = cyaw[idx]
        self._last_ind_list = ind_list   # store for border-constraint lookup
        return ref_traj

    ################################################################
    def _linear_mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.opti.set_value(self.x0k, x0)
        NX, NU, T = self.config.NXK, self.config.NU, self.config.TK
        A_bd = np.zeros((NX * T, NX * T))
        B_bd = np.zeros((NX * T, NU * T))
        C_bd = np.zeros(NX * T)
        for t in range(T):
            delta_bar = self.odelta_v[t] if self.odelta_v is not None else 0.0
            A, B, C = self._get_model_matrix(
                path_predict[2, t], path_predict[3, t], delta_bar)
            A_bd[t*NX:(t+1)*NX, t*NX:(t+1)*NX] = A
            B_bd[t*NX:(t+1)*NX, t*NU:(t+1)*NU] = B
            C_bd[t*NX:(t+1)*NX] = C
        self.opti.set_value(self.Ak_param, A_bd)
        self.opti.set_value(self.Bk_param, B_bd)
        self.opti.set_value(self.Ck_param, C_bd)

        #Border constraints lookup : last target_ind and the pre-loaded border coefficients.
        if self._has_border and self._last_ind_list is not None:
            bc = self.border_coeffs[self._last_ind_list]   # (T+1, 4)
            self.opti.set_value(self.border_a_k,  bc[:, 0])
            self.opti.set_value(self.border_b_k,  bc[:, 1])
            self.opti.set_value(self.border_cl_k, bc[:, 2])
            self.opti.set_value(self.border_cr_k, bc[:, 3])

        ref_traj[3, :] = x0[3] + wrap_angle(ref_traj[3, :] - x0[3])
        for i in range(1, ref_traj.shape[1]):
            d = ref_traj[3, i] - ref_traj[3, i - 1]
            if d >  np.pi: ref_traj[3, i] -= 2*np.pi
            if d < -np.pi: ref_traj[3, i] += 2*np.pi
        self.opti.set_value(self.ref_traj_k, ref_traj)

        oa     = self.oa       if self.oa       is not None else np.zeros(self.config.TK)
        odelta = self.odelta_v if self.odelta_v is not None else np.zeros(self.config.TK)

        NX, NU, T = self.config.NXK, self.config.NU, self.config.TK
        if self._prev_xk is not None and self._prev_uk is not None:
            x_init = np.zeros((NX, T+1)); u_init = np.zeros((NU, T))
            x_init[:, :-1] = self._prev_xk[:, 1:]; x_init[:, -1] = self._prev_xk[:, -1]
            u_init[:, :-1] = self._prev_uk[:, 1:]; u_init[:, -1] = self._prev_uk[:, -1]
            self.opti.set_initial(self.xk, x_init)
            self.opti.set_initial(self.uk, u_init)
        else:
            self.opti.set_initial(self.xk, ref_traj)
            self.opti.set_initial(self.uk, np.zeros((NU, T)))

        try:
            sol    = self.opti.solve()
            oa     = np.array(sol.value(self.uk[0, :])).flatten()
            odelta = np.array(sol.value(self.uk[1, :])).flatten()
            self._prev_xk = np.array(sol.value(self.xk))
            self._prev_uk = np.array(sol.value(self.uk))
        except Exception as e:
            try:
                oa     = np.array(self.opti.debug.value(self.uk[0, :])).flatten()
                odelta = np.array(self.opti.debug.value(self.uk[1, :])).flatten()
                self._prev_xk = np.array(self.opti.debug.value(self.xk))
                self._prev_uk = np.array(self.opti.debug.value(self.uk))
            except Exception:
                pass   # carry-forward oa/odelta already set above
        return oa, odelta

    ##########################################################################
    def _linear_mpc_control(self, ref_path, x0):
        oa = self.oa       if self.oa       is not None else [0.0]*self.config.TK
        od = self.odelta_v if self.odelta_v is not None else [0.0]*self.config.TK
        path_predict = self._predict_motion(x0, oa, od, ref_path)
        return self._linear_mpc_prob_solve(ref_path, path_predict, x0)

    #########################################################################
    def update_yaw(self, yaw_raw: float) -> float:
        if self.prev_odom_yaw is None:
            self.prev_odom_yaw = yaw_raw
        dyaw = yaw_raw - self.prev_odom_yaw
        if dyaw >  np.pi: self.yaw_offset -= 2*np.pi
        if dyaw < -np.pi: self.yaw_offset += 2*np.pi
        self.prev_odom_yaw = yaw_raw
        return yaw_raw + self.yaw_offset

# Gym runner
#########################################################################################

class GymMPCRunner:
    
    def __init__(self, waypoints_csv: str, map_name: str = "Spielberg",
                 border_coeffs_csv: str = None):
        self.mpc      = HeadlessMPC(waypoints_csv, border_coeffs_csv=border_coeffs_csv)
        self.map_name = map_name
        self._log: list    = []   # list of dicts, one per control step
        self._frames: list = []   # annotated RGB frames
        
        # mutable state shared between run-loop and render callbacks
        self._driven_state = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}
        self._pred_state   = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}

    def run(self,
            start_x: float = 0.0,
            start_y: float = 0.0,
            start_theta: float = 0.0,
            max_sim_seconds: float = 60.0):

        import gymnasium as gym
        from f1tenth_gym.envs.observation import ObservationType
        from f1tenth_gym.envs.dynamic_models import DynamicModel
        from f1tenth_gym.envs.integrators import IntegratorType
        from f1tenth_gym.envs.reset import ResetStrategy
        from f1tenth_gym.envs.env_config import (
            ControlConfig, EnvConfig, ObservationConfig, ResetConfig, SimulationConfig,
        )

        env_cfg = EnvConfig(
            map_name=self.map_name,
            observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
            simulation_config=SimulationConfig(
                timestep=0.01,
                integrator_timestep=0.01,
                integrator=IntegratorType.RK4,
                dynamics_model=DynamicModel.KS,
                compute_frenet_frame=False,
                max_laps=None,
            ),
            control_config=ControlConfig(steer_delay_steps=1),
            reset_config=ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC),
        )
        env = gym.make(
            'f1tenth_gym:f1tenth-v0',
            config=env_cfg,
            render_mode="rgb_array",
        )

        try:
            track = env.unwrapped.track
            poses = np.array([[
                track.raceline.xs[0],
                track.raceline.ys[0],
                track.raceline.yaws[0],
            ]])
        except Exception:
            poses = np.array([[start_x, start_y, start_theta]])
        obs, info = env.reset(options={"poses": poses})


        # Reset per-run overlay state and register render callbacks
        self._driven_state = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}
        self._pred_state   = {'renderer': None, 'points': np.zeros((1, 2), dtype=np.float32)}
        self._driven_pts_list: list = []
        self._setup_render_callbacks(env)

        cfg      = self.mpc.config
        ref_x    = self.mpc.waypoints[:, 0]
        ref_y    = self.mpc.waypoints[:, 1]
        ref_yaw  = self.mpc.waypoints[:, 2]
        ref_v    = calc_speed_profile(ref_x, ref_y, ref_yaw)

      
        GYM_DT        = 0.01
        MPC_INTERVAL  = max(1, int(cfg.DTK / GYM_DT))
        steer         = 0.0

        # speed_desired is the integrated velocity target sent to the gym.
        # Seeded from the reference profile so the car moves before the first
        # successful IPOPT solve; thereafter updated by MPC acceleration output.

        speed_desired = float(ref_v[0])
        mpc_valid     = False          # True once we have a real IPOPT solution
        ref_path      = None           # persisted across MPC intervals
        sim_time      = 0.0
        step_count    = 0
        t_wall_start   = time.time()
       
        RENDER_EVERY   = max(1, int(1.0 / GYM_DT))   # 1 frame / sim-second
        self._frames   = []
        self._log      = []

        pred_horizon_x: list = []   # MPC horizon x
        pred_horizon_y: list = []   # MPC horizon y

        # Live per-second snapshot display
        try:
            from IPython import get_ipython as _gip
            _IN_COLAB = _gip() is not None
        except Exception:
            _IN_COLAB = False
        _last_preview_sec = -1

        DEBUG_EVERY = max(1, int(1.0 / GYM_DT))   # print once per sim-second
        print(f"Running MPC in gym for up to {max_sim_seconds:.0f}s sim-time ...")
        print(f"{'step':>7}  {'sim_t':>7}  {'x':>8}  {'y':>8}  {'v_obs':>7}  {'v_cmd':>7}  {'steer_deg':>10}  {'CTE':>8}")

        try:
            while sim_time < max_sim_seconds:
                
                agent_id = list(obs.keys())[0]
                agent_obs = obs[agent_id]

                yaw_raw = float(agent_obs['pose_theta'])
                yaw_c   = self.mpc.update_yaw(yaw_raw)
                vx      = float(agent_obs.get('linear_vel_x', 0.0))
                vy      = float(agent_obs.get('linear_vel_y', 0.0))
                v       = math.sqrt(vx**2 + vy**2)

                vehicle_state = State(
                    x   = float(agent_obs['pose_x']),
                    y   = float(agent_obs['pose_y']),
                    v   = v,
                    yaw = yaw_c,
                )

               
                if step_count % MPC_INTERVAL == 0:
                    ref_path = self.mpc.calc_ref_trajectory(
                        vehicle_state, ref_x, ref_y, ref_yaw, ref_v
                    )
                    x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
                    self.mpc.oa, self.mpc.odelta_v = self.mpc._linear_mpc_control(ref_path, x0)

                    # Cache predicted horizon for visualisation
                    if self.mpc._prev_xk is not None:
                        pred_horizon_x = self.mpc._prev_xk[0, :].tolist()
                        pred_horizon_y = self.mpc._prev_xk[1, :].tolist()

                    # Steering: from MPC odelta_v[0]
                    if self.mpc.odelta_v is not None:
                        steer = float(np.clip(self.mpc.odelta_v[0],
                                              cfg.MIN_STEER, cfg.MAX_STEER))

                    
                    # Speed: integrate MPC acceleration into a desired velocity.
                    # The gym KS model tracks a target velocity — so we maintain
                    # speed_desired and update it with oa[0] each MPC interval.
                    # On a failed solve (oa all-zero first time) we fall back to
                    # the reference speed profile to keep the car moving.
                    if self.mpc.oa is not None and mpc_valid:
                        speed_desired = float(np.clip(
                            v + self.mpc.oa[0] * cfg.DTK,
                            cfg.MIN_SPEED, cfg.MAX_SPEED))
                        mpc_valid = True
                    else:
                        # Reference speed at the current look-ahead point
                        speed_desired = float(np.clip(
                            ref_path[2, 1], cfg.MIN_SPEED, cfg.MAX_SPEED))
                        # Promote to MPC-driven once oa is non-trivial
                        if (self.mpc.oa is not None
                                and np.any(np.abs(self.mpc.oa) > 1e-3)):
                            mpc_valid = True

                # Signed CTE
                ind  = self.mpc.target_ind
                cx_  = ref_x[ind]; cy_ = ref_y[ind]; cy_yaw = ref_yaw[ind]
                cte  = (-(vehicle_state.x - cx_) * math.sin(cy_yaw)
                        + (vehicle_state.y - cy_) * math.cos(cy_yaw))
                self._log.append({
                    't':        sim_time,
                    'x':        vehicle_state.x,
                    'y':        vehicle_state.y,
                    'yaw':      yaw_raw,          # Uncorrected raw yaw from visualization
                    'v':        v,
                    'speed_cmd': speed_desired,
                    'steer':    steer,
                    'cte':      cte,
                    'pred_x':   list(pred_horizon_x),
                    'pred_y':   list(pred_horizon_y),
                })

                action = np.array([[steer, speed_desired]])
                obs, step_reward, done, truncated, info = env.step(action)

                # Update driven-path overlay
                self._driven_pts_list.append(
                    [vehicle_state.x, vehicle_state.y])

                # Capture one annotated gym frame per simulated second
                if step_count % RENDER_EVERY == 0:
                    frame = env.render()          # GL render with callbacks baked in
                    frame = self._hud_overlay(    # PIL text HUD on top
                        frame, sim_time, v, steer, cte)
                    self._frames.append(frame)


                    # Live snapshot
                    cur_sec = int(sim_time)
                    if _IN_COLAB and cur_sec > _last_preview_sec:
                        _last_preview_sec = cur_sec
                        try:
                            from IPython.display import display, Image as _Img
                            from io import BytesIO
                            _buf = BytesIO()
                            from PIL import Image as _PILImg
                            _PILImg.fromarray(frame).save(_buf, format='PNG')
                            _buf.seek(0)
                            from IPython.display import clear_output
                            clear_output(wait=True)
                            display(_Img(data=_buf.read()))
                        except Exception:
                            pass

                # Debug print once per sim-second with state and MPC info
                if step_count % DEBUG_EVERY == 0:
                    print(f"{step_count:>7}  {sim_time:>7.2f}s  "
                          f"{vehicle_state.x:>8.3f}  {vehicle_state.y:>8.3f}  "
                          f"{v:>7.3f}  {speed_desired:>7.3f}  "
                          f"{math.degrees(steer):>10.2f}  {cte:>8.4f}")

                sim_time   += GYM_DT
                step_count += 1

                # Stop on collision
                collided = any(done.values()) if isinstance(done, dict) else bool(done)
                if collided:
                    print(f"[COLLISION] step={step_count}  sim_t={sim_time:.3f}s  "
                          f"x={vehicle_state.x:.3f}  y={vehicle_state.y:.3f}  "
                          f"v={v:.3f} m/s")
                    break

        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] step={step_count}  sim_t={sim_time:.3f}s  "
                  f"frames={len(self._frames)}  log-entries={len(self._log)}")
            print("Saving captured frames before exit ...")

        env.close()
        wall = time.time() - t_wall_start
        collided_str = "  COLLISION" if (any(done.values()) if isinstance(done, dict) else bool(done)) else ""
        print(f"Simulation finished.  sim-time={sim_time:.2f}s  wall-time={wall:.1f}s  "
              f"steps={step_count}  log-entries={len(self._log)}{collided_str}")

    #####################################################################################
    def _setup_render_callbacks(self, env):
        """
        Registers line + point renderers inside the gym's GL pipeline onto every frame.
        """
        wp_x = self.mpc.waypoints[:, 0]
        wp_y = self.mpc.waypoints[:, 1]
        raceline_pts = np.column_stack([wp_x, wp_y]).astype(np.float32)

        has_border = (self.mpc._has_border
                      and hasattr(self.mpc, 'border_walls_xy')
                      and self.mpc.border_walls_xy is not None)

        # Static geometry: created once on first callback invocation
        static = {'done': False}

        def _static_cb(er):
            if static['done']:
                return
            static['done'] = True
            
            er.get_lines_renderer(raceline_pts, color=(255, 255, 255), size=1)
            if has_border:
                bw = self.mpc.border_walls_xy
                er.get_lines_renderer(
                    np.column_stack([bw[:, 0], bw[:, 1]]).astype(np.float32),
                    color=(255, 215, 0), size=2)   # left wall
                er.get_lines_renderer(
                    np.column_stack([bw[:, 2], bw[:, 3]]).astype(np.float32),
                    color=(255, 107, 53), size=2)  # right wall
                bc = self.mpc.border_corridor_xy
                er.get_lines_renderer(
                    np.column_stack([bc[:, 0], bc[:, 1]]).astype(np.float32),
                    color=(200, 170, 0), size=1)   # left corridor
                er.get_lines_renderer(
                    np.column_stack([bc[:, 2], bc[:, 3]]).astype(np.float32),
                    color=(200, 90, 30), size=1)   # right corridor

        # Currently they remain static at the last updated position; but they need to be made dynamically updating
        # Driven trail - updated every frame
        # Capture self directly; read _driven_pts_list at callback time (no stale alias)
        runner = self

        def _driven_cb(er):
            pts_list = runner._driven_pts_list
            if len(pts_list) < 2:
                return
            pts = np.array(pts_list, dtype=np.float32)
            if runner._driven_state['renderer'] is None:
                runner._driven_state['renderer'] = er.get_lines_renderer(
                    pts, color=(0, 207, 255), size=2)   #Driven path
            else:
                runner._driven_state['renderer'].update(pts)

        # MPC predicted horizon 
        def _pred_cb(er):
            xk = runner.mpc._prev_xk
            if xk is None:
                return
            pts = np.column_stack([xk[0, :], xk[1, :]]).astype(np.float32)
            if len(pts) < 2:
                return
            if runner._pred_state['renderer'] is None:
                runner._pred_state['renderer'] = er.get_lines_renderer(
                    pts, color=(57, 255, 20), size=2)   #MPC predicted horizon
            else:
                runner._pred_state['renderer'].update(pts)

        env.unwrapped.add_render_callback(_static_cb)
        env.unwrapped.add_render_callback(_driven_cb)
        env.unwrapped.add_render_callback(_pred_cb)

    ###########################################################################
    @staticmethod
    def _hud_overlay(frame: np.ndarray, sim_t, v, steer, cte) -> np.ndarray:
        """Telemetry text heads up display onto the numpy frame array using PIL."""
        from PIL import Image, ImageDraw
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        lines = [
            (f"t  = {sim_t:.1f} s",      (255, 255, 255)),
            (f"v  = {v:.2f} m/s",        (0,   207, 255)),
            (f"\u03b4  = {math.degrees(steer):.1f}\u00b0", (255, 215,   0)),
            (f"CTE= {cte:+.3f} m",       (255, 107,  53)),
        ]
        x0, y0, dy = 10, 10, 18
        for i, (text, colour) in enumerate(lines):
            # thin shadow for readability
            draw.text((x0 + 1, y0 + i*dy + 1), text, fill=(0, 0, 0))
            draw.text((x0,     y0 + i*dy),     text, fill=colour)
        return np.asarray(img)

    ###########################################################################
    def save_mp4(self, out_path: str = "mpc_run.mp4", fps: int = 5):
        """Write the annotated gym frames captured during run() to .mp4."""
        if not self._frames:
            print("No frames captured "); return
        
        import imageio
        print(f"Writing {len(self._frames)} annotated frames → {out_path} ...")
        imageio.mimwrite(out_path, self._frames, fps=fps, quality=8)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Saved: {out_path}  ({size_mb:.1f} MB)")

    #########################################################################
    def plot_cte(self, out_path: str = "cte_plot.png"):
        if not self._log:
            print("No log data."); return

        ts   = [e['t']       for e in self._log]
        ctes = [e['cte'] for e in self._log]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ts, ctes, color='#e94560', linewidth=1)
        ax.set_xlabel('sim time [s]'); ax.set_ylabel('CTE [m]')
        ax.set_title(f'Cross-Track Error  (mean={np.mean(ctes):.4f} m  '
                     f'max={np.max(ctes):.4f} m)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"CTE plot saved: {out_path}")


###################################################################################################################
def colab_main():
    
    print("Starting final Constrained-MPC runner in Colab ...")
    # Files should be in the same directory as the notebook
    runner = GymMPCRunner(
        waypoints_csv     = "waypoints.csv",
        map_name          = "Spielberg",            
        border_coeffs_csv = "Spielberg_border_coeffs.csv",
    )
    try:
        runner.run(
            start_x      = 0.0,
            start_y      = 0.0,
            start_theta  = 0.0,
            max_sim_seconds = 60.0*12,  # 12 minutes sim-time max
        )
    finally:
        # 1 frame/s captured → play at 5 fps = 5× speed summary
        runner.save_mp4("mpc_run.mp4", fps=5)
        runner.plot_cte("cte_plot.png")

        try:
            from IPython.display import display, Video, Image
            print("\n=== Final video ===")
            display(Video("mpc_run.mp4", embed=True, width=800))
            print("\n=== CTE plot ===")
            display(Image("cte_plot.png"))
        except Exception:
            pass   


if __name__ == "__main__":
    colab_main()
