import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def circular_trajectory(t, center=(0.,0.,0.), R=1.0, omega=1.0, phi0=0.0, vz=0.0):
    """
    Circular trajectory in world frame (XY circle, optional Z climb).
    
    Args:
        t: scalar or array of times
        center: (cx, cy, cz), circle center
        R: radius
        omega: angular speed (rad/s)
        phi0: initial phase
        vz: vertical climb rate (m/s)
    
    Returns:
        pos, vel, acc, yaw
        Each shape (...,3) except yaw (...,)
    """
    cx, cy, cz = center
    theta = omega * t + phi0

    # Position
    x = cx + R * jnp.cos(theta)
    y = cy + R * jnp.sin(theta)
    z = cz + vz * t
    pos = jnp.stack([x, y, z], axis=-1)

    # Velocity
    vx = -R * omega * jnp.sin(theta)
    vy =  R * omega * jnp.cos(theta)
    vz_arr = jnp.broadcast_to(vz, jnp.shape(theta))
    vel = jnp.stack([vx, vy, vz_arr], axis=-1)

    # Acceleration
    ax = -R * omega**2 * jnp.cos(theta)
    ay = -R * omega**2 * jnp.sin(theta)
    az = jnp.zeros_like(theta)
    acc = jnp.stack([ax, ay, az], axis=-1)

    # Yaw (velocity direction)
    yaw = jnp.arctan2(vy, vx)

    return pos, vel, acc, yaw


if __name__ == "__main__":
    t = jnp.linspace(0, 10, 1000)
    pos, vel, acc, yaw = circular_trajectory(
        t, center=(1.,2.,0.), R=1.5, omega=2.094, phi0=0.0, vz=0.1
    )
    print(pos[0,0])
    print(pos[0,1])
    print(pos[0,2])
    plt.figure(figsize=(6,6))
    plt.plot(pos[:,0], pos[:,1], label="Trajectory (XY)")
    plt.quiver(pos[::50,0], pos[::50,1], vel[::50,0], vel[::50,1],
               angles="xy", scale_units="xy", scale=5, color="r", label="Velocity")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Circular Trajectory with Velocity Vectors")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    # save figure
    plt.savefig("circular_trajectory.png", dpi=300)
    plt.show()

    print("Figure saved as circular_trajectory.png")