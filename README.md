# Kalman Filter Tests
Just checking out how a Kalman Filter works.

## Dependencies
- numpy
- scipy
- matplotlib

### Installing
I ran into an error trying to install scipy in a requirements.txt. To get arond this, I installed numpy and scipy separately before installing pylab.
```sh
$ pip install numpy==1.10.4
$ pip install scipy==0.17.0
$ pip install pylab==0.1.3
```

## Example usage
```sh
$ python test_voltmeter.py  # Test noisy voltmeter
$ eog voltmeter.png  # View results
```


## Samples
![voltmeter](/images/voltmeter.png)
![cannon](/images/cannon.png)


## Resources
- http://greg.czerniak.info/guides/kalman1/
- http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mjx-eqn-kalpredictfull
