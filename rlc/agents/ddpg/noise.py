import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(
        self,
        action_dimension: int,
        theta: float = 0.15,
        sigma: float = 0.2,
        mu: float = 0,
        dt: float = 1,
    ) -> None:
        self.action_dimension = action_dimension
        self.mu = np.ones(self.action_dimension) * mu

        self.dt = dt
        self.theta = theta
        self.sigma = sigma

        self.state = self.mu.copy()
        self.rng = np.random.default_rng()

    def reset(self) -> None:
        self.state = self.mu.copy()

    def step(self) -> np.ndarray:
        drift = self.theta * (self.mu - self.state) * self.dt
        noise = self.rng.normal(loc=self.mu, size=self.action_dimension)
        diffusion = self.sigma * noise
        self.state += drift + diffusion
        return self.state.copy()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    from matplotlib.animation import FuncAnimation

    noise = OrnsteinUhlenbeckProcess(action_dimension=3, theta=0.05, sigma=0.1, mu=0,)
    frames = 10000

    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    data = np.array([noise.step() for index in range(frames)]).T

    (line,) = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])

    def update(frame):
        i = frame - 100 if frame > 100 else 0
        line.set_data(data[0:2, i:frame])
        line.set_3d_properties(data[2, i:frame])
        return (line,)

    def init():
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_xlabel("X")

        ax.set_ylim3d([-1.0, 1.0])
        ax.set_ylabel("Y")

        ax.set_zlim3d([-1.0, 1.0])
        ax.set_zlabel("Z")

        ax.set_title("3D Test")
        return (line,)

    ani = FuncAnimation(fig, update, frames, interval=10, init_func=init)

    plt.show()
