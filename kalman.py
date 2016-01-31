#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy


class KalmanFilterLinear(object):
    """
    Class modelling a linear Kalman Filter.
    Linear because all the matrices used are constant and do not change
    over time.

    Equations
    State prediciton:
        Get our prediction of the next state given the last state (x_last) and
        control vector (u). The control vector represents tracked external
        influences to the system unrelated to the state itself
        (ex: gravity/wind acting on a ball that was thrown.)
        x_predicted = A * x_last + B * u
    Covariance prediction:
        Get our predicted error of the next state of the system given the last
        state error (P_last) and estimated error covariance (Q). Q is
        essentially the spread of noise in a system caused by untrcked
        influences.
        P_predicted = A * P_last * A_transposed + Q
    Innovation:
        Get how much different our prediction of the next state (x_predicted)
        is from the measurement from sensors (z_n).
        y = z_n - H * x_predicted
    Innovation covariance:
        Get predicted measurement error covariance given the last state error
        covariance (P_predicted) and estimated measurement error
        covariance (R). This is essentially the uncertainty caused by
        errors in the sensors.
        S = H * P_predicted * H_transposed + R
    Kalman gain:
        How much the state should be changed from the differences in
        measurement given the predicted covariance (P_predicted) and
        innovation covariance (S).
        K = P_predicted * H_transposed * S_inverse
    State update:
        Update the next estimate of the state from the prediction
        (x_predicted), the Kalman Gain (K), and innovation (y).
        x = x_predicted + K * y
    Covariance update:
        Update the next estimate of the error from the prediction
        (P_predicted) and the Kalman Gain (K)
        P = (I - K * H) * P_predicted

    Matrices (these are provided beforehand)
    A:
        State transition matrix.
        Matrix for predicting the next state of a system in the next time step
        from the previous state.
    B:
        Control matrix.
        Matrix for predicting the next state of a system from a control vector.
    H:
        Observation matrix.
        Matrix used for converting state values to measurement values.
        For example, if we want to keep track of our position, velocity, and
        acceleration, but our measurements are lat/lng coords, to compare
        against the lat/lng measured by GPS, we will need to convert our
        predicted pos, vel, and acc to lat/lng.
        (H: pos, vel, acc => lat, lng)
    Q:
        Estimated process error covariance.
        Covariance from untracked external influences outside of the system.
    R:
        Estimated process error covariance.
        Covariance from uncertainty in the sensors getting the measurements.
    """

    def __init__(self, A, B, H, x_init, P_init, Q, R):
        self._A = A  # State transition matrix
        self._A_transposed = numpy.transpose(A)
        self._B = B  # Control matrix
        self._H = H  # Observation matrix
        self._H_transposed = numpy.transpose(H)
        self._x = x_init  # State estimate
        self._P = P_init  # Covariance estimate
        self._Q = Q  # Estimated error in process
        self._R = R  # Estimated error in measurements

    @property
    def state_estimate(self):
        """Current state of the system (x)."""
        return self._x

    def update(self, control, measurement):
        """
        Update the state + covariance from a new control vector
        and measurement vector.
        """
        # Prediction
        x_predicted = self._A * self._x + self._B * control
        P_predicted = self._A * self._P * self._A_transposed + self._Q

        # Observation
        innovation = measurement - self._H * x_predicted
        innovation_P = self._H * P_predicted * self._H_transposed + self._R

        # Update
        K = P_predicted * self._H_transposed * numpy.linalg.inv(innovation_P)
        self._x = x_predicted + K * innovation
        size = K.shape[0]
        self._P = (numpy.eye(size) - K * self._H) * P_predicted

