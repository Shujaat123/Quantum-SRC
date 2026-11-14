import json,tempfile,unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from qsrc.quantum import states,fidelity
from qsrc.solver import solve
from qsrc.data import prepare
from qsrc.data_extra import GRID_FEATURES,parse_grid,occlude
from qsrc.tables import reproduce

ROOT=Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_fidelity_is_projector_inner_product(self):
        S=states(np.array([[.1,.3],[.4,.8],[.9,1.2]]),qubits=3,depth=2)
        rho=np.array([np.outer(v,v.conj()) for v in S])
        explicit=np.einsum('aij,bji->ab',rho,rho).real
        np.testing.assert_allclose(fidelity(S),explicit,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(fidelity(S)).min(),-1e-12)

    def test_terminal_cnot_cancels_at_depth_one(self):
        X=np.random.default_rng(3).uniform(0,np.pi,(5,8))
        A=fidelity(states(X,qubits=6,depth=1,entangle=True))
        B=fidelity(states(X,qubits=6,depth=1,entangle=False))
        np.testing.assert_allclose(A,B,atol=1e-12)

    def test_elastic_net_kkt(self):
        K=np.array([[1.,.3],[.3,1.]]);k=np.array([.8,.2])
        a,info=solve(K,k,.05,.1)
        self.assertLess(info['kkt_residual'],1e-6)
        self.assertTrue(np.isfinite(a).all())

    def test_training_dictionary_is_disjoint_from_validation_and_test(self):
        d=prepare('wine',2,6,10,8,12)
        sets=[set(d[k]) for k in ['dictionary_ids','validation_ids','test_ids']]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertTrue(sets[0]<=set(d['training_ids']))

    def test_grid_targets_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'grid.csv'
            p.write_text(','.join(GRID_FEATURES+['stab','stabf'])+'\n'+','.join(['1']*12+['999999','stable'])+'\n')
            X,y=parse_grid(p);self.assertEqual(X.shape,(1,12));self.assertTrue(np.all(X==1));self.assertEqual(y.tolist(),[1])

    def test_occlusion_actual_area_and_seed(self):
        X=np.ones((3,4096));A,area=occlude(X,.1,7);B,_=occlude(X,.1,7)
        self.assertEqual(area,400/4096);np.testing.assert_array_equal(A,B)
        np.testing.assert_array_equal((A==0).sum(axis=1),[400]*3)
        self.assertTrue(np.all(X==1))

    def test_all_archived_metrics_and_tables(self):
        with tempfile.TemporaryDirectory() as out:
            result=reproduce(ROOT/'archived_results',Path(out))
            self.assertEqual(int(result.n_seeds.sum()),240)
            self.assertEqual(len(result),24)
            import pandas as pd
            self.assertEqual(len(pd.read_csv(Path(out)/'finite_shot_results.csv')),1920)
            self.assertEqual(len(pd.read_csv(Path(out)/'per_seed_results.csv')),320)
            src=result[(result.dataset=='faces')&(result.model=='quantum_src')].iloc[0]
            self.assertAlmostEqual(src.mean_balanced_accuracy,.7975)

    def test_corrupted_metric_is_rejected(self):
        from qsrc.tables import read_results
        with tempfile.TemporaryDirectory() as out:
            out=Path(out);(out/'manifest.json').write_text('{}')
            p=out/'runs/iris_seed0/result.json';p.parent.mkdir(parents=True)
            case=json.loads((ROOT/'archived_results/runs/iris_seed0/result.json').read_text())
            case['exact'][0]['accuracy']=.12345;p.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError,'disagrees'):read_results(out)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
