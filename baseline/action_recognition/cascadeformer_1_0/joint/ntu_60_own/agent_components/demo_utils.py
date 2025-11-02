import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .constants import NTU25_EDGES


def _get_limits(window, margin_ratio=0.1, use_3d=False):
    """Compute plot limits with a small margin from data."""
    if use_3d and window.shape[-1] >= 3:
        mins = window.min(axis=(0, 1))
        maxs = window.max(axis=(0, 1))
        span = np.maximum(maxs - mins, 1e-6)
        mins -= span * margin_ratio
        maxs += span * margin_ratio
        return (mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])
    else:
        xy = window[..., :2]
        mins = xy.min(axis=(0, 1))
        maxs = xy.max(axis=(0, 1))
        span = np.maximum(maxs - mins, 1e-6)
        mins -= span * margin_ratio
        maxs += span * margin_ratio
        return (mins[0], maxs[0]), (mins[1], maxs[1])

def make_skeleton_video(
    window: np.ndarray,
    out_path: str,
    fps: int = 12,
    edges=NTU25_EDGES,
    title: str = "Skeleton Demo",
    use_3d: bool = True
):
    """
    Render a skeleton-only video from `window` of shape (T, 25, C).
    If use_3d=False, renders 2D using (x,y). If True and C>=3, renders 3D.
    """
    T, V, C = window.shape
    assert V == 25, f"Expected 25 joints; got {V}"
    assert C >= 2, "Need at least 2 channels (x,y)."

    # Matplotlib setup
    fig = plt.figure(figsize=(5, 5), dpi=200)
    if use_3d and C >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection='3d')
        (xlim, ylim, zlim) = _get_limits(window, use_3d=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        (xlim, ylim) = _get_limits(window, use_3d=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.invert_yaxis()

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if use_3d and C >= 3:
        ax.set_zticklabels([])

    ax.set_title(title)
    ax.set_aspect("equal")

    # Initialize artists: joints + bones
    if use_3d and C >= 3:
        joint_scatter = ax.scatter([], [], [], s=10)
        bone_lines = [ax.plot([], [], [], linewidth=1.5)[0] for _ in edges]
    else:
        joint_scatter = ax.scatter([], [], s=10)
        bone_lines = [ax.plot([], [], linewidth=1.5)[0] for _ in edges]

    def init():
        if use_3d and C >= 3:
            joint_scatter._offsets3d = ([], [], [])
            for ln in bone_lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
        else:
            joint_scatter.set_offsets(np.empty((0, 2)))
            for ln in bone_lines:
                ln.set_data([], [])
        return [joint_scatter, *bone_lines]

    def update(t):
        pts = window[t]
        if use_3d and C >= 3:
            xs, ys, zs = pts[:, 0], pts[:, 2], pts[:, 1]  # swap y <-> z
            joint_scatter._offsets3d = (xs, ys, zs)
            for ln, (i, j) in zip(bone_lines, edges):
                ln.set_data([xs[i], xs[j]], [ys[i], ys[j]])
                ln.set_3d_properties([zs[i], zs[j]])
        else:
            xs, ys = pts[:, 0], pts[:, 1]
            joint_scatter.set_offsets(np.stack([xs, ys], axis=1))
            for ln, (i, j) in zip(bone_lines, edges):
                ln.set_data([xs[i], xs[j]], [ys[i], ys[j]])
        return [joint_scatter, *bone_lines]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        blit=not (use_3d and C >= 3), interval=1000/fps
    )

    try:
        if animation.writers.is_available("ffmpeg"):
            ani.save(out_path, writer="ffmpeg", fps=fps, dpi=200, bitrate=2000)
            print(f"Saved {out_path}")
        elif animation.writers.is_available("pillow"):
            gif_path = out_path.replace(".mp4", ".gif")
            ani.save(gif_path, writer="pillow", fps=fps)
            print(f"ffmpeg not found; saved GIF instead: {gif_path}")
        else:
            print("No suitable writer found. Install ffmpeg or pillow to save the animation.")
    finally:
        plt.close(fig)