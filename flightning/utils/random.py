import chex
import jax.random as jrandom
import jax.numpy as jnp
import jax.scipy.spatial.transform as transform


def key_generator(seed):
    """
    Generator for random keys. Use for debugging and testing only!
    >>> key_gen = key_generator(0)
    >>> key1 = next(key_gen)
    >>> key2 = next(key_gen)
    """
    key = jrandom.key(seed)
    while True:
        key, subkey = jrandom.split(key)
        yield subkey


def random_rotation_matrix(key: jnp.ndarray) -> jnp.ndarray:
    """Generates a random rotation matrix."""
    random_vec = jrandom.normal(key, (3,))
    return transform.Rotation.from_rotvec(random_vec).as_matrix()


def random_rotation(
    key: chex.PRNGKey, yaw_scale: float, pitch_scale: float, roll_scale: float
) -> transform.Rotation:
    key_yaw, key_pitch, key_roll = jrandom.split(key, 3)
    yaw = yaw_scale * jrandom.uniform(
        key_yaw, minval=-jnp.pi, maxval=jnp.pi
    )
    pitch = pitch_scale * jrandom.uniform(
        key_pitch, minval=-jnp.pi, maxval=jnp.pi
    )
    roll = roll_scale * jrandom.uniform(
        key_roll, minval=-jnp.pi, maxval=jnp.pi
    )
    # convert to rotation matrix (assuming extrinsic rotations)
    rotation = transform.Rotation.from_euler(
        "zyx", jnp.array([yaw, pitch, roll])
    )
    return rotation

def rotation_with_random(
    key: chex.PRNGKey, yaw_scale: float, pitch_scale: float, roll_scale: float, yaw_init: float, pitch_init: float, roll_init: float,
) -> transform.Rotation:
    key_yaw, key_pitch, key_roll = jrandom.split(key, 3)
    yaw_rand = yaw_scale * jrandom.uniform(
        key_yaw, minval=-jnp.pi, maxval=jnp.pi
    )
    pitch_rand = pitch_scale * jrandom.uniform(
        key_pitch, minval=-jnp.pi, maxval=jnp.pi
    )
    roll_rand = roll_scale * jrandom.uniform(
        key_roll, minval=-jnp.pi, maxval=jnp.pi
    )
    # convert to rotation matrix (assuming extrinsic rotations)
    yaw = yaw_init + yaw_rand
    pitch = pitch_init + pitch_rand
    roll = roll_init + roll_rand
    rotation = transform.Rotation.from_euler(
        "zyx", jnp.array([yaw, pitch, roll])
    )
    return rotation
