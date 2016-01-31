#!/usr/bin/env python
# -*_ coding: utf-8 -*-

import random
import sys
import numpy
import matplotlib.pylab as pylab

from kalman import KalmanFilterLinear


class Voltmeter(object):
    """Measure voltage with a noisy voltmeter."""

    def __init__(self, voltage, noise):
        self._voltage = voltage
        self._noise = noise

    @property
    def voltage(self):
        return self._voltage

    @property
    def noisy_voltage(self):
        return random.gauss(self.voltage, self._noise)


def test():
    """Test measuring a constant voltage from a voltmeter."""
    pylab.clf()  # Clear the plot
    steps = 60

    # 1 b/c we are measuring a constant voltage source, so
    # the next expected voltage should be the exact same as the last.
    # Just multiply by 1.
    A = numpy.matrix([1])

    # 1 b/c measuring voltage directly.
    # (The state is in the same units as the measurement.)
    H = numpy.matrix([1])

    # 0 b/c no outside forces affect the state.
    B = numpy.matrix([0])

    # Add some small random covariance.
    Q = numpy.matrix([0.00001])

    # Random covariance from measurment.
    R = numpy.matrix([0.1])

    # Initial state/covariance
    x_init = numpy.matrix([3])
    P_init = numpy.matrix([1])

    kf = KalmanFilterLinear(A, B, H, x_init, P_init, Q, R)

    # The voltage we expect is 1.25 V and we have given an initial guess
    # of 3 V. Let's see how good this filter is.
    voltmeter = Voltmeter(1.25, 0.25)

    measured_voltage = []
    true_voltage = []
    kalman_voltage = []

    for i in xrange(steps):
        measured = voltmeter.noisy_voltage
        measured_voltage.append(measured)
        true_voltage.append(voltmeter.voltage)
        kalman_voltage.append(kf.state_estimate[0, 0])
        kf.update(numpy.matrix([0]), numpy.matrix([measured]))

    pylab.plot(xrange(steps), measured_voltage, 'b',
               xrange(steps), true_voltage, 'r',
               xrange(steps), kalman_voltage, 'g')
    pylab.xlabel('Time')
    pylab.ylabel('Voltage')
    pylab.title('Voltage Measurement with Kalman Filter')
    pylab.legend(('measured', 'true voltage', 'kalman'))
    pylab.savefig('voltmeter.png')


if __name__ == "__main__":
    sys.exit(test())

