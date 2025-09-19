#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute path curvature, surface roughness, and jerk from Odometry.

Assumptions
-----------
- We treat planimetric motion (x, y) only for curvature/roughness.
- Jerk is computed from speed magnitudes assuming unit time step (Δt = 1).
  If your Odometry is not uniformly sampled, replace finite differences
  with time-stamped derivatives using header.stamp.

Outputs
-------
- Prints average curvature, normalized curvature, roughness, and jerk once
  when x-position first exceeds a user-defined threshold.

Author: (your name)
"""

import numpy as np
import warnings
import rospy
from nav_msgs.msg import Odometry
from typing import List, Tuple

# -------------------------
# Global state (kept minimal)
# -------------------------
poses_xy: List[Tuple[float, float]] = []   # list of (x, y)
vel_xy:   List[Tuple[float, float]] = []   # list of (vx, vy)

did_compute_once = False                   # guard to run metrics once
X_THRESHOLD = 27.0                         # trigger to compute metrics


# -------------------------
# Geometry helpers (2D)
# -------------------------
def triangle_area_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Unsigned area of triangle ABC in 2D."""
    # 0.5 * | (b - a) x (c - a) |  where cross is the scalar z-component
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0.5 * abs(cross)


def calc_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float]:
    """
    Curvature from 3 points via circumcircle formula: k = 4A / (|AB||BC||CA|).
    Returns (curvature, normalized_curvature) where the latter scales by local arc.
    """
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    # Degenerate guards
    if ab == 0.0 or bc == 0.0 or ca == 0.0:
        return 0.0, 0.0

    area = triangle_area_2d(a, b, c)
    denom = ab * bc * ca
    if denom == 0.0:
        return 0.0, 0.0

    k = 4.0 * area / denom  # curvature
    # A common "normalized" flavor is scaling by local arc length proxy
    k_norm = k * (ab + bc)
    return k, k_norm


def calc_roughness(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    A simple roughness proxy using triangle area and chord length.
    Here we use: roughness = 2 * Area(ABC) / |AC|^2
    (dimensionless measure; larger => 'rougher' turns/undulations).
    """
    area = triangle_area_2d(a, b, c)
    chord_sq = np.sum((c - a) ** 2)
    if chord_sq == 0.0:
        return 0.0
    return (2.0 * area) / chord_sq


def calc_jerk(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> Tuple[float, float]:
    """
    Compute jerk magnitude from three consecutive velocity vectors, assuming Δt = 1.
    - speed magnitudes: s1, s2, s3
    - 'acceleration' (finite difference on speed): a1 = s2 - s1, a2 = s3 - s2
    - jerk = |a2 - a1|
    Returns (jerk, a1)
    """
    s1 = float(np.linalg.norm(v1))
    s2 = float(np.linalg.norm(v2))
    s3 = float(np.linalg.norm(v3))
    a1 = s2 - s1
    a2 = s3 - s2
    return abs(a2 - a1), a1


# -------------------------
# Batch metrics
# -------------------------
def get_curvatures(points: List[Tuple[float, float]]) -> Tuple[List[float], List[float]]:
    """Compute curvature and normalized curvature over sliding triples."""
    n = len(points)
    if n < 3:
        return [], []

    curvatures, curvatures_norm = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n - 2):
            a = np.array(points[i], dtype=float)
            b = np.array(points[i + 1], dtype=float)
            c = np.array(points[i + 2], dtype=float)
            k, k_norm = calc_curvature(a, b, c)
            curvatures.append(k)
            curvatures_norm.append(k_norm)
    return curvatures, curvatures_norm


def get_roughnesses(points: List[Tuple[float, float]]) -> List[float]:
    """Compute roughness over sliding triples."""
    n = len(points)
    if n < 3:
        return []
    vals = []
    for i in range(n - 2):
        a = np.array(points[i], dtype=float)
        b = np.array(points[i + 1], dtype=float)
        c = np.array(points[i + 2], dtype=float)
        vals.append(calc_roughness(a, b, c))
    return vals


def get_jerks(velocities: List[Tuple[float, float]]) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute jerk list, acceleration-on-speed list, and speed magnitudes.
    """
    m = len(velocities)
    if m < 3:
        return [], [], []

    jerks, accs = [], []
    for i in range(m - 2):
        v1 = np.array(velocities[i], dtype=float)
        v2 = np.array(velocities[i + 1], dtype=float)
        v3 = np.array(velocities[i + 2], dtype=float)
        j, a1 = calc_jerk(v1, v2, v3)
        jerks.append(j)
        accs.append(a1)

    speeds = [float(np.linalg.norm(v)) for v in np.asarray(velocities, dtype=float)]
    return jerks, accs, speeds


def safe_mean(values: List[float]) -> float:
    """Mean that returns 0.0 for empty input."""
    return float(np.mean(values)) if len(values) > 0 else 0.0


# -------------------------
# ROS callback
# -------------------------
def odom_callback(msg: Odometry) -> None:
    """
    Odometry subscriber callback.
    Collect (x,y) position and (vx,vy) linear velocity. When x > X_THRESHOLD
    for the first time, compute and print the metrics once.
    """
    global did_compute_once

    # Extract position (planar) and linear velocity
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    # NOTE: use position.z for altitude if needed; orientation.z is a quaternion part.
    # z = msg.pose.pose.position.z

    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y

    poses_xy.append((x, y))
    vel_xy.append((vx, vy))

    if (not did_compute_once) and (x > X_THRESHOLD):
        did_compute_once = True
        rospy.loginfo("Threshold crossed (x > %.3f). Computing metrics...", X_THRESHOLD)

        curv, curv_n = get_curvatures(poses_xy)
        rough = get_roughnesses(poses_xy)
        jerk, acc, speeds = get_jerks(vel_xy)

        rospy.loginfo("Curvature (mean):             %.6f", safe_mean(curv))
        rospy.loginfo("Curvature (normalized, mean): %.6f", safe_mean(curv_n))
        rospy.loginfo("Roughness (mean):             %.6f", safe_mean(rough))
        rospy.loginfo("Jerk (mean):                  %.6f", safe_mean(jerk))
        rospy.loginfo("Samples -> pose: %d, vel: %d", len(poses_xy), len(vel_xy))


# -------------------------
# Main
# -------------------------
def main() -> None:
    rospy.init_node("path_metrics_node", anonymous=False)
    rospy.Subscriber("/odom", Odometry, odom_callback, queue_size=50)
    rospy.loginfo("path_metrics_node started. Waiting for /odom...")
    rospy.spin()


if __name__ == "__main__":
    main()
