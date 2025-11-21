import numpy as np



def low_level_controller_numpy(omega_current, f_T, omega_cmd,
                               allocation_matrix_inv,
                               inertial_matrix,
                               thrust_min, thrust_max,
                               motor_omega_min, motor_omega_max,
                               thrust_map):
    """
    Low-level controller using NumPy.

    Args:
        state: object or dict with 'omega' (3D vector)
        f_T: total thrust command (scalar)
        omega_cmd: desired body angular rates (3D vector)
        allocation_matrix_inv: inverse of allocation matrix (4x4)
        inertial_matrix: 3x3 moment of inertia matrix
        thrust_min, thrust_max: min/max thrust per motor
        motor_omega_min, motor_omega_max: min/max motor speeds
        thrust_map: thrust = k * omega^2 mapping (e.g. array([k, ...]))

    Returns:
        motor_omega_d: desired motor angular velocities (4D vector)
    """

    # P-gain matrix for body rate control
    K = np.diag([20.0, 20.0, 41.0])


    # Compute body rate error
    omega_err = omega_cmd - omega_current

    # Compute commanded torques
    body_torques_cmd = inertial_matrix @ (K @ omega_err) + np.cross(
        omega_current, inertial_matrix @ omega_current
    )

    # Combine thrust and torques
    alpha = np.zeros(4)
    alpha[0] = f_T[0]
    alpha[1:4] = body_torques_cmd
    # alpha = np.concatenate(([f_T], body_torques_cmd))

    # Compute individual motor thrusts
    f_cmd = allocation_matrix_inv @ alpha

    # Clamp thrusts
    f_cmd = np.clip(f_cmd, thrust_min, thrust_max)

    # Convert to desired motor speeds (thrust = k * ω²)
    motor_omega_d = np.sqrt(f_cmd / thrust_map[0])

    # Clamp motor speeds
    motor_omega_d = np.clip(motor_omega_d, motor_omega_min, motor_omega_max)

    return motor_omega_d


if __name__=="__main__":
    inertial_matrix = np.array([[0.00027226, 0.        , 0.        ],
       [0.        , 0.00027226, 0.        ],
       [0.        , 0.        , 0.0003704 ]])
    
    omega_cmd = np.array([0.0, 0.0, 1.0])
    omega_current = np.zeros(3)
    f_T = np.array([9.81])
    allocation_matrix = np.array([[ 1.   ,  1.   ,  1.   ,  1.   ],
       [-0.04 ,  0.04 , -0.04 ,  0.04 ],
       [-0.04 ,  0.04 ,  0.04 , -0.04 ],
       [-0.008, -0.008,  0.008,  0.008]])
    thrust_min = np.array([0.0045])
    thrust_max = np.array([3.5])
    motor_omega_min = np.array([150])
    motor_omega_max = np.array([4400])
    thrust_map = np.array([2.e-07, 0.e+00, 0.e+00])
    allocation_matrix_inv = np.linalg.inv(allocation_matrix)
    low_level_controller_numpy(omega_current, f_T, omega_cmd,
                               allocation_matrix_inv,
                               inertial_matrix,
                               thrust_min, thrust_max,
                               motor_omega_min, motor_omega_max,
                               thrust_map)