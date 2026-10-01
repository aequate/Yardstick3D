import numpy as np

def test_rms_vector_triangle_bound_with_correlated_sensor_bias():
    rng=np.random.default_rng(37)
    true=rng.normal(size=(8,3))
    predicted=true+rng.normal(size=(8,3))*.7
    # Constant systematic bias, deliberately not IID zero-mean noise.
    measured=true+np.array([1.,-.3,.4])
    rms=lambda x:np.sqrt(np.mean(np.sum(x*x,axis=1)))
    residual=rms(measured-predicted)
    sensor=rms(measured-true)
    error=rms(true-predicted)
    assert max(0.,residual-sensor)-1e-12<=error<=residual+sensor+1e-12

def test_horizontal_agreement_cannot_bound_vertical_error():
    true=np.column_stack([np.arange(8),np.zeros(8),np.zeros(8)])
    predicted=true.copy();predicted[:,2]=100*np.sin(np.arange(8))
    np.testing.assert_array_equal(true[:,:2],predicted[:,:2])
    assert np.sqrt(np.mean(np.sum((true-predicted)**2,axis=1)))>50
