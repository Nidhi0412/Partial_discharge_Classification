import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from training_safety import (ResourceLimitError, check_resources, recovery_callback,
                             snapshot_path, training_lock)


class SafetyTests(unittest.TestCase):
    def test_low_memory_and_disk_refuse_training(self):
        with patch('training_safety.memory_available', return_value=.5), \
             patch('training_safety.shutil.disk_usage', return_value=SimpleNamespace(free=100*2**30)):
            with self.assertRaises(ResourceLimitError): check_resources({}, '.')
        with patch('training_safety.memory_available', return_value=8), \
             patch('training_safety.shutil.disk_usage', return_value=SimpleNamespace(free=2**30)):
            with self.assertRaises(ResourceLimitError): check_resources({}, '.')

    def test_lock_rejects_concurrent_process_access(self):
        with tempfile.TemporaryDirectory() as d:
            with training_lock(Path(d)/'lock'):
                with self.assertRaises(RuntimeError):
                    with training_lock(Path(d)/'lock'): pass
            with training_lock(Path(d)/'lock'): pass

    def test_checkpoint_paths_cannot_escape_run(self):
        for name in ['../epoch_0001.keras', 'best.keras', '/epoch_0001.keras']:
            with self.assertRaises(ValueError): snapshot_path('.', name)

    def test_epoch_state_preserves_patience_history_and_best_across_resume(self):
        tf = SimpleNamespace(keras=SimpleNamespace(callbacks=SimpleNamespace(Callback=object)))
        model = SimpleNamespace(save=lambda path: Path(path).write_bytes(b'generated test checkpoint'), stop_training=False)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); config=dict(epochs=10,patience=2)
            cb=recovery_callback(tf,out,config); cb.model=model
            cb.on_epoch_end(0, {'loss':1.,'val_loss':.5})
            cb.on_epoch_end(1, {'loss':1.,'val_loss':.4})
            self.assertFalse((out/'epoch_0001.keras').exists())
            state=json.loads((out/'recovery_state.json').read_text())
            # Increase LR patience independently so this test needs no TF optimizer.
            config['patience']=6
            cb=recovery_callback(tf,out,config,state); cb.model=model
            cb.on_epoch_end(2, {'loss':1.,'val_loss':.6})
            self.assertEqual(cb.state['wait'],1)
            self.assertEqual(cb.state['best_file'],'epoch_0002.keras')
            self.assertEqual(len(cb.state['history']['val_loss']),3)
            self.assertTrue((out/'epoch_0002.keras').exists())
            self.assertTrue((out/'epoch_0003.keras').exists())

    def test_nonfinite_epoch_does_not_commit(self):
        tf=SimpleNamespace(keras=SimpleNamespace(callbacks=SimpleNamespace(Callback=object)))
        with tempfile.TemporaryDirectory() as d:
            cb=recovery_callback(tf,Path(d),dict(epochs=2,patience=2))
            with self.assertRaises(FloatingPointError): cb.on_epoch_end(0,{'val_loss':float('nan')})
            self.assertFalse((Path(d)/'recovery_state.json').exists())

    def test_stopping_and_learning_rate_continue_after_restart(self):
        class Rate:
            value=1.
            def assign(self,value): self.value=value
        rate=Rate()
        tf=SimpleNamespace(keras=SimpleNamespace(callbacks=SimpleNamespace(Callback=object),
                            backend=SimpleNamespace(get_value=lambda value:value.value)))
        model=SimpleNamespace(save=lambda path:Path(path).write_bytes(b'checkpoint'),
                              optimizer=SimpleNamespace(learning_rate=rate),stop_training=False)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); config=dict(epochs=10,patience=2)
            cb=recovery_callback(tf,out,config);cb.model=model
            cb.on_epoch_end(0,{'val_loss':.5})
            cb.on_epoch_end(1,{'val_loss':.6})
            self.assertEqual(rate.value,.5)
            state=json.loads((out/'recovery_state.json').read_text())
            cb=recovery_callback(tf,out,config,state);cb.model=model
            cb.on_epoch_end(2,{'val_loss':.7})
            self.assertTrue(model.stop_training)
            self.assertTrue(cb.state['finished'])
            self.assertEqual(cb.state['completed_epochs'],3)
            self.assertEqual(cb.state['best_file'],'epoch_0001.keras')
            self.assertEqual(rate.value,.25)

    def test_cli_resume_skips_completed_folds_and_keeps_all_predictions(self):
        import training_pipeline as pipeline
        calls=[]
        rows=[dict(image_id=str(i),image_path=f'image{i}',recording_id=str(i),group_id=str(i),
                   material='Nomex' if i%2==0 else 'Teflon',fold=str(1+i//2),partition='validation')
              for i in range(4)]
        def train(model,train,val,config,out):
            calls.append(out.name)
            if len(calls)==2: raise KeyboardInterrupt('test interruption in second fold')
            checkpoint=out/'best.keras'; checkpoint.write_bytes(b'generated fake model')
            return object(),checkpoint
        def predict(model,val,*args):
            return [dict(r,y_true=i%2,p_teflon=.1 if i%2==0 else .9,y_pred=i%2) for i,r in enumerate(val)]
        tf=SimpleNamespace(__version__='test',keras=SimpleNamespace(backend=SimpleNamespace(clear_session=lambda:None)))
        with tempfile.TemporaryDirectory() as d, \
             patch.object(pipeline,'load_splits',return_value=(rows,dict(split_sha256='test',group_by='group_id',folds=2))), \
             patch.object(pipeline,'tensorflow',return_value=tf), \
             patch.object(pipeline,'configure_runtime'), \
             patch.object(pipeline,'check_resources',return_value={}), \
             patch.object(pipeline,'train_one',side_effect=train), \
             patch.object(pipeline,'predict_rows',side_effect=predict):
            out=Path(d)/'run'
            with self.assertRaises(KeyboardInterrupt):
                pipeline.main(['cv','--splits',str(Path(d)/'splits.csv'),'--output-dir',str(out)])
            self.assertTrue((out/'fold_1/completed.json').exists())
            pipeline.main(['resume','--run-dir',str(out)])
            self.assertEqual(calls,['fold_1','fold_2','fold_2'])
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'COMPLETE')
            self.assertEqual(json.loads((out/'metrics.json').read_text())['image']['samples'],4)


if __name__=='__main__': unittest.main()
