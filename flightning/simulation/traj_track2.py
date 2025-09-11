import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import PoseStamped


def generate_circular_trajectory(
    radius=2.0,
    altitude=1.5,
    speed=1.0,
    duration=20.0,
    dt=0.02,
    yaw_mode="tangent",
    phi0=0.0,
    mass=1.0,
    g=9.81,
):
    """
    Generate a circular trajectory and quadcopter states along it.

    Returns a pandas DataFrame with columns:
      t, x, y, z, vx, vy, vz, ax, ay, az,
      yaw, yaw_rate, qw, qx, qy, qz,
      thrust_N, roll, pitch
    """

    omega = speed / radius  # angular speed (rad/s)
    times = np.arange(0.0, duration + 1e-9, dt)
    rows = []

    for t in times:
        theta = omega * t + phi0

        # Position
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = altitude

        # Velocity
        vx = -radius * omega * np.sin(theta)
        vy = radius * omega * np.cos(theta)
        vz = 0.0

        # Acceleration (centripetal)
        ax = -radius * omega**2 * np.cos(theta)
        ay = -radius * omega**2 * np.sin(theta)
        az = 0.0

        # Yaw & yaw rate
        if yaw_mode == "tangent":
            yaw = np.arctan2(vy, vx)
            denom = vx * vx + vy * vy
            yaw_rate = (vx * ay - vy * ax) / denom if denom > 1e-8 else 0.0
        elif yaw_mode == "fixed":
            yaw, yaw_rate = 0.0, 0.0
        elif yaw_mode == "inward":
            yaw, yaw_rate = np.arctan2(-y, -x), 0.0
        else:
            yaw, yaw_rate = 0.0, 0.0

        # Desired acceleration (world frame, incl. gravity)
        a_des = np.array([ax, ay, az + g])
        thrust_N = np.linalg.norm(a_des) * mass

        # Orientation (body z-axis aligned with thrust)
        zb = a_des / np.linalg.norm(a_des)
        x_yaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        yb = np.cross(zb, x_yaw)
        if np.linalg.norm(yb) < 1e-8:
            yb = np.cross(zb, [0, 0, 1]) if abs(zb[2]) < 0.99 else np.cross(zb, [0, 1, 0])
        yb /= np.linalg.norm(yb)
        xb = np.cross(yb, zb)
        xb /= np.linalg.norm(xb)

        R_bw = np.column_stack((xb, yb, zb))
        rot = R.from_matrix(R_bw)
        qx, qy, qz, qw = rot.as_quat()  # scipy returns [x,y,z,w]

        roll, pitch, _ = rot.as_euler("xyz", degrees=False)
        print(pitch)

        rows.append(
            [t, x, y, z, vx, vy, vz, ax, ay, az, yaw, yaw_rate,
             qw, qx, qy, qz, thrust_N, roll, pitch]
        )

    cols = [
        "t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az",
        "yaw", "yaw_rate", "qw", "qx", "qy", "qz", "thrust_N", "roll", "pitch"
    ]

    return pd.DataFrame(rows, columns=cols)


def plot_trajectory(df, step=20, scale=0.3):
    """Plot the 3D trajectory, XY projection, and orientation arrows."""
    fig = plt.figure(figsize=(12, 6))

    # 3D plot
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(df["x"], df["y"], df["z"], label="Trajectory")

    # Add orientation arrows every 'step' frames
    for i in range(0, len(df), step):
        x, y, z = df.loc[i, ["x", "y", "z"]]
        q = [df.loc[i, "qx"], df.loc[i, "qy"], df.loc[i, "qz"], df.loc[i, "qw"]]
        rot = R.from_quat(q)
        R_mat = rot.as_matrix()
        # Body axes
        xb, yb, zb = R_mat[:, 0], R_mat[:, 1], R_mat[:, 2]

    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.set_title("3D Trajectory with Orientation")
    ax1.legend()

    # XY projection
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(df["x"], df["y"], label="XY Projection")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_title("Top View (XY)")
    ax2.axis("equal")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    """Generate and plot a sample circular trajectory with orientations."""
    df = generate_circular_trajectory(
        radius=1.5,
        altitude=1.0,
        speed=1.0,
        duration=10.0,
        dt=0.02,
        yaw_mode="fixed",
        mass=0.234,
    )
    print(df.head())
    plot_trajectory(df)
    df.to_csv("trajectory.csv", index=False) 

    return df

class TrajTrack(object):
    def __init__(self, traj):
        self.pose_pub = rospy.Publisher("/pose_pub", PoseStamped, queue_size=5)
        self.timer = rospy.Timer(rospy.Duration(0.05), self.traj_pub)
        self.full_traj = traj
        self.counter = 0
        self.size = df.shape[0]

    def traj_pub(self, event):
        posey = PoseStamped()
        posey.header.frame_id = "map"
        posey.pose.position.x = self.full_traj["x"][self.counter]
        posey.pose.position.y = self.full_traj["y"][self.counter]
        posey.pose.position.z = self.full_traj["z"][self.counter]
        posey.pose.orientation.x = self.full_traj["qx"][self.counter]
        posey.pose.orientation.y = self.full_traj["qy"][self.counter]
        posey.pose.orientation.z = self.full_traj["qz"][self.counter]
        posey.pose.orientation.w = self.full_traj["qw"][self.counter]
        self.pose_pub.publish(posey)
        self.counter += 1
        if self.counter % self.size == 0:
            self.counter = 0


if __name__ == "__main__":
    df = main()
    rospy.init_node("test")
    trajtrack = TrajTrack(df)
    rospy.spin()

    