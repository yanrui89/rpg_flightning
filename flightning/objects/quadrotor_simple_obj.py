import jax
import jax_dataclasses as jdc
from jax import numpy as jnp

from flightning.utils.math import rotation_matrix_from_vector
from flightning.utils.pytrees import field_jnp, CustomPyTree


@jdc.pytree_dataclass
class QuadrotorSimpleState(CustomPyTree):
    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])


def quadrotor_dyn(
        p: jax.Array,
        R: jax.Array,
        v: jax.Array,
        a: jax.Array,
        omega: jax.Array,
        dt: jax.Array,
        gravity: jax.Array = jnp.array([0, 0, -9.81]),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Quadrotor dynamics model.
    :param p: position
    :param R: orientation matrix
    :param v: velocity
    :param a: acceleration
    :param omega: body rates
    :param dt: time step
    :param gravity: gravity vector
    :return: new position, orientation matrix, and velocity
    """
    # Euler step for position and velocity
    p_new = p + dt * v
    v_new = v + dt * (gravity + R @ jnp.array([0, 0, a]))
    # Exact step for orientation
    R_delta = rotation_matrix_from_vector(dt * omega)
    R_new = R @ R_delta
    return p_new, R_new, v_new

def quadrotor_dyn_with_res(
        p: jax.Array,
        R: jax.Array,
        v: jax.Array,
        a: jax.Array,
        res_acc: jax.Array,
        omega: jax.Array,
        dt: jax.Array,
        gravity: jax.Array = jnp.array([0, 0, -9.81]),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Quadrotor dynamics model.
    :param p: position
    :param R: orientation matrix
    :param v: velocity
    :param a: acceleration
    :param omega: body rates
    :param dt: time step
    :param gravity: gravity vector
    :return: new position, orientation matrix, and velocity
    """
    # Euler step for position and velocity
    p_new = p + dt * v
    nominal_acc = gravity + R @ jnp.array([0, 0, a])
    corrected_acc = nominal_acc + res_acc
    v_new = v + dt * corrected_acc
    # Exact step for orientation
    R_delta = rotation_matrix_from_vector(dt * omega)
    R_new = R @ R_delta
    return p_new, R_new, v_new


class QuadrotorSimple:
    """
    Simplified quadrotor model. No low-level control or aerodynamic effects.
    """

    def __init__(self, mass=0.752):
        """
        :param mass: quadrotor mass in kg
        """
        self.mass = mass

    def step(
            self,
            state: QuadrotorSimpleState,
            f: jax.Array,
            omega: jax.Array,
            dt: jax.Array,
            gravity: jax.Array = jnp.array([0, 0, -9.81]),
    ) -> QuadrotorSimpleState:
        """
        :param state: quadrotor state
        :param f: cumulative thrust
        :param omega: body rates
        :param dt: time step length in s
        :param gravity: gravity vector
        :return: next state of the quadrotor
        """
        a = f / self.mass
        p, R, v = quadrotor_dyn(
            state.p, state.R, state.v, a, omega, dt, gravity
        )
        # noinspection PyArgumentList
        state.replace(p=p, R=R, v=v)
        return state

class QuadrotorSimplewithRes:
    """
    Simplified quadrotor model. No low-level control or aerodynamic effects.
    """

    def __init__(self, mass=0.752):
        """
        :param mass: quadrotor mass in kg
        """
        self.mass = mass

        #Residual model - Loading torch into flax and then freeze params


    def step(
            self,
            state: QuadrotorSimpleState,
            f: jax.Array,
            omega: jax.Array,
            dt: jax.Array,
            gravity: jax.Array = jnp.array([0, 0, -9.81]),
    ) -> QuadrotorSimpleState:
        """
        :param state: quadrotor state
        :param f: cumulative thrust
        :param omega: body rates
        :param dt: time step length in s
        :param gravity: gravity vector
        :return: next state of the quadrotor
        """
        a = f / self.mass
        inputs = jnp.concatenate([
                state.p.reshape(-1),          # (3,)
                state.R.reshape(-1),          # (9,)
                state.v.reshape(-1),          # (3,)
                jnp.array([a]),         # (1,)
                omega.reshape(-1),      # (3,)
            ])
        # residual_world_acc = self.flax_model.apply(self.params_flax, inputs)
        p, R, v = quadrotor_dyn_with_res(
            state.p, state.R, state.v, a, omega, dt, gravity
        )
        # noinspection PyArgumentList
        state.replace(p=p, R=R, v=v)
        return state




def main():
    # Initialize quadrotor state
    state = QuadrotorSimpleState(
    p=jnp.array([1.0, -2.0, 0.5]),          # some offset in space
    v=jnp.array([0.5, -0.3, 0.2]),          # initial velocity
    R=jnp.array([[0.0, -1.0, 0.0],          # rotated 90 deg around z
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]])
)

    # Create quadrotor model
    quad = QuadrotorSimple(mass=0.752)
    resstate = QuadrotorSimplewithRes(mass = 0.752)

    # Define inputs for one step
    f = 7.0                 # total thrust in N
    omega = jnp.array([5.1, 0.2, 0.3])  # body rates in rad/s
    dt = 0.01               # time step in seconds

    # Perform one dynamics step
    for i in range(100):
        state = quad.step(state, f=f, omega=omega, dt=dt)
    # new_res_state = quad.step(state, f=f, omega=omega, dt=dt)

    # Print results
    print("Old position:", state.p)
    # print("New position:", new_state.p)
    # print("New res position:", new_res_state.p)
    print("Old velocity:", state.v)
    # # print("New velocity:", new_state.v)
    # print("New res velocity:", new_res_state.v)
    print("Old rotation matrix:\n", state.R)
    # print("New rotation matrix:\n", new_state.R)
    # print("New res rotation matrix:\n", new_res_state.R)

        


if __name__ == "__main__":
    main()