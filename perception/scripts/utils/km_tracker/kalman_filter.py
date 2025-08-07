import numpy as np

class KalmanFilter:
    def __init__(self, dt ,future_step, init_state, process_noise_std, measurement_noise_std):
        # Initial State
        self.state =  init_state  # [x, y, vx, vy]
        self.covariance = np.eye(4)
        self.n = future_step
        self.future_states = []
        for _ in range(self.n):
            self.future_states.append(init_state)

        
        # State Transition Model
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement Model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process Noise
        self.Q = np.eye(4) * process_noise_std
        
        # Measurement Noise
        self.R = np.eye(2) * measurement_noise_std
        
    def predict(self):
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.last_state = np.array([self.state[0],self.state[1]])
        
    def update(self, measurement,flag):
        if flag :
            new_data = measurement
        else:
            new_data = self.last_state

            
        K = self.covariance @ self.H.T @ np.linalg.inv(self.H @ self.covariance @ self.H.T + self.R)
        self.state = self.state + K @ (new_data - self.H @ self.state)
        self.covariance = (np.eye(4) - K @ self.H) @ self.covariance
        return np.array([[self.state[0]], [self.state[1]]]) 

    def _predict_future_states(self):

        future_states = []
        state = self.state.copy()
        for _ in range(self.n):
            state = self.F @ state
            future_states.append(state.copy())
        self.future_states = future_states.copy()


if __name__ == '__main__':
    # Example Usage
    dt = 0.1    # dt 表示时间间隔
    kf = KalmanFilter(dt,5, np.array([0, 0, 0, 0]), 0.1, 0.1)

    # Suppose we have some measurements
    measurements = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4]),
                    np.array([5, 5]), np.array([6, 6]), np.array([7, 7]), np.array([8, 8])
                    ]

    for measurement in measurements:
        kf.predict()
        kf.update(measurement,True)
        print("State:", kf.state)
        # print("Covariance:", kf.covariance, "\n")
        print("Future States:", kf.future_states, "\n")