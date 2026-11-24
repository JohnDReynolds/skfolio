"""Independent new-coordinate oracles; original public assertions remain unchanged."""
from decimal import Decimal,localcontext
import numpy as np
import pandas as pd
import scipy.optimize as sco
from skfolio.prior import BasePrior,ReturnDistribution,EntropyPooling
from skfolio.optimization import MeanRisk


def square():
    return pd.DataFrame([[.01,.01],[.01,-.01],[-.01,.01],[-.01,-.01]],columns=['A','B'])


def test_normalized_cost_gradient_and_public_posterior(monkeypatch):
    X=square();before=X.copy(deep=True);real=sco.minimize;checked=[]
    # Derived normalized columns: correlation, mean A, mean B, two fixed variances.
    centered=np.array([[.5,.5,.5,0,0],[-.5,.5,-.5,0,0],[-.5,-.5,.5,0,0],[.5,-.5,-.5,0,0]])
    def inspect(fun,x0,**kwargs):
        assert x0.shape==(5,)
        assert kwargs['bounds']==[(None,None)]+[(-1000,1000)]*4
        points=[np.zeros(5),np.array([1e-10,0,0,0,0]),np.array([.2,-.3,.4,.1,-.2]),np.array([1e200,0,0,0,0])]
        for point in points:
            with localcontext() as ctx:
                ctx.prec=260
                logits=[-sum(Decimal(str(v))*Decimal(str(y)) for v,y in zip(row,point)) for row in centered]
                maximum=max(logits)
                factors=[(v-maximum).exp() if v-maximum>Decimal(-10000) else Decimal(0) for v in logits]
                total=sum(factors);prob=[v/total for v in factors]
                expected=float(maximum+(total/Decimal(4)).ln())
                gradient=[float(-sum(Decimal(str(centered[i,j]))*prob[i] for i in range(4))) for j in range(5)]
            with np.errstate(over='raise',invalid='raise'):
                value,grad=fun(point)
            np.testing.assert_allclose(value,expected,rtol=3e-12,atol=3e-14)
            np.testing.assert_allclose(grad,gradient,rtol=3e-12,atol=3e-14)
        checked.append(True)
        return real(fun,x0=x0,**kwargs)
    monkeypatch.setattr(sco,'minimize',inspect)
    model=EntropyPooling(correlation_views=['(A, B) == 0']).fit(X)
    assert checked
    np.testing.assert_allclose(model.return_distribution_.sample_weight,[.25]*4,atol=1e-9,rtol=0)
    pd.testing.assert_frame_equal(X,before)


def test_full_length_readonly_options_and_complete_public_result(monkeypatch):
    X=square();before=X.copy(deep=True);scale=np.array([12345.,1,1,1,1,1]);offset=np.array([6789.,0,0,0,0,0]);scale.setflags(write=False);offset.setflags(write=False)
    params={'maxfun':5000,'ftol':1e-11,'xtol':1e-8,'gtol':1e-8,'stepmx':1,'scale':scale,'offset':offset};real=sco.minimize;checked=[]
    def inspect(fun,x0,**kwargs):
        assert x0.shape==(5,)
        options=kwargs['options'];assert options is not params
        np.testing.assert_array_equal(options['scale'],[1]*5)
        np.testing.assert_array_equal(options['offset'],[0]*5)
        assert not np.shares_memory(options['scale'],scale)
        for key in ['maxfun','ftol','xtol','gtol','stepmx']:assert options[key]==params[key]
        checked.append(True)
        return real(fun,x0=x0,**kwargs)
    monkeypatch.setattr(sco,'minimize',inspect)
    model=EntropyPooling(correlation_views=['(A, B) == .3'],solver_params=params).fit(X)
    expected=np.array([.325,.175,.175,.325]);cov=1e-4*np.array([[1,.3],[.3,1]])/(1-expected@expected)
    np.testing.assert_allclose(model.return_distribution_.sample_weight,expected,atol=1.5e-5/4,rtol=0)
    np.testing.assert_allclose(model.return_distribution_.covariance,cov,atol=5e-9,rtol=0)
    np.testing.assert_almost_equal(model.relative_entropy_,np.sum(expected*np.log(4*expected)),5)
    opt=MeanRisk(prior_estimator=model).fit(X)
    np.testing.assert_allclose(opt.weights_,[.5,.5],atol=1e-6,rtol=0)
    np.testing.assert_allclose(opt.predict(X).returns,[.01,0,0,-.01],atol=2e-8,rtol=0)
    assert checked and model.solver_params is params
    assert params['scale'] is scale and params['offset'] is offset
    np.testing.assert_array_equal(scale,[12345,1,1,1,1,1]);np.testing.assert_array_equal(offset,[6789,0,0,0,0,0]);assert not scale.flags.writeable and not offset.flags.writeable
    pd.testing.assert_frame_equal(X,before)


class FixedPrior(BasePrior):
    def __init__(self,sample_weight):self.sample_weight=sample_weight
    def fit(self,X,y=None):
        self.return_distribution_=ReturnDistribution(mu=np.average(X,axis=0,weights=self.sample_weight),covariance=np.cov(X,rowvar=False,aweights=self.sample_weight),returns=np.asarray(X),sample_weight=self.sample_weight)
        return self


def test_preserve_nonzero_mean_public_control():
    X=pd.DataFrame([[.03,.01],[-.01,.02],[-.01,-.02],[-.01,0]],columns=['A','B']);before=X.copy(deep=True);prior=np.array([.01,.33,.33,.33]);prior.setflags(write=False)
    model=EntropyPooling(prior_estimator=FixedPrior(prior),mean_views=['A == .029'],correlation_views=['(A, B) >= -1']).fit(X)
    expected=np.array([.975,1/120,1/120,1/120]);mean=np.array([.029,.00975]);centered=np.asarray(X)-mean;cov=(centered.T*expected)@centered/(1-expected@expected)
    np.testing.assert_allclose(model.return_distribution_.sample_weight,expected,atol=3e-6,rtol=0)
    np.testing.assert_allclose(model.return_distribution_.mu,mean,atol=5e-8,rtol=0)
    np.testing.assert_allclose(model.return_distribution_.covariance,cov,atol=5e-9,rtol=0)
    pd.testing.assert_frame_equal(X,before);np.testing.assert_array_equal(prior,[.01,.33,.33,.33]);assert not prior.flags.writeable
