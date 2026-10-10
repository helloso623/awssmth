"""Host mathematics tests; model training executes only on Trainium."""
import math,unittest
import numpy as np
import torch
from objective import adaptive_beta,entropic_advantages,importance_loss,token_advantages

class ObjectiveTests(unittest.TestCase):
    def test_beta_and_loo(self):
        rewards=[0.,.8,1.,1.2]
        beta,status=adaptive_beta(rewards)
        self.assertEqual(status,'attained')
        z=np.exp(beta*(np.asarray(rewards)-max(rewards)))
        q=z/z.sum()
        self.assertAlmostEqual(float((q*np.log(4*q)).sum()),math.log(2),places=10)
        a,_=entropic_advantages(rewards)
        expected=z/((z.sum()-z)/3+1e-12)-1
        np.testing.assert_allclose(a.numpy(),expected,rtol=2e-5,atol=2e-5)
    def test_equal_and_unattainable(self):
        a,info=entropic_advantages([0.,0.,0.,0.])
        self.assertEqual(info['beta'],0.)
        self.assertTrue(torch.equal(a,torch.zeros(4)))
        _,status=adaptive_beta([1.,1.,1.,0.])
        self.assertEqual(status,'unattainable_or_cap_limited')
    def test_kl_and_prompt_mask(self):
        mask=[torch.tensor([0.,1.,1.])]
        sample=[torch.tensor([float('inf'),-.3,-.7])]
        ref=[torch.tensor([0.,-.4,-.5])]
        # Ignore nonfinite masked prompt metadata before KL centering.
        finite_sample=[torch.tensor([0.,-.3,-.7])]
        adv,average=token_advantages([2.],finite_sample,ref,mask)
        self.assertAlmostEqual(average,-.05,places=6)
        p=torch.tensor([5.,-.2,-.6],requires_grad=True)
        loss=importance_loss([p],sample,adv,mask)
        expected=-(torch.exp(p[1:]-sample[0][1:])*adv[0][1:]).sum()
        torch.testing.assert_close(loss,expected)
        loss.backward();self.assertEqual(float(p.grad[0]),0.)

if __name__=='__main__':unittest.main()
