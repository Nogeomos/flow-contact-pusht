"""Recover command provenance after an offline/online log filename collision.

Old evidence is preserved. All four learned policies are rerun in fresh processes
without changing checkpoints, simulator, controller, test seeds, or action budget.
The recovery records compare every saved NPZ array and distinguish coverage's tiny
GEOS floating-point summation differences from actual physical trajectory changes.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.run_experiments import run


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'runs/stages234_20261022'
    destination=base/'fresh_verification'
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir()
    configs=[('contact','contact_main','contact_test','learned_contact',0),
             ('flowaction','flow_action','flow_action_test','flow_action',0),
             ('seed1','contact_seed1','contact_seed1_test','learned_contact',1),
             ('seed2','contact_seed2','contact_seed2_test','learned_contact',2)]
    frozen=json.loads((base/'controller_freeze_v2.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'research/controller.py')==frozen['controller_sha256']
    assert sha(ROOT/'research/evaluate_closed_loop.py')==frozen['evaluator_sha256']
    observations=[]
    before={}
    for name,model,old,method,seed in configs:
        for path in sorted((base/old).glob('*')):
            if path.is_file():before[str(path.relative_to(ROOT))]=sha(path)
        checkpoint=base/model/'best.pt'
        before[str(checkpoint.relative_to(ROOT))]=sha(checkpoint)
        old_log=base/'logs'/f'{old}.command.json'
        old_record=json.loads(old_log.read_text(encoding='utf-8')) if old_log.exists() else None
        copied_log=base/'logs'/f'closed_loop_{old}.command.json'
        copied_record=json.loads(copied_log.read_text(encoding='utf-8')) if copied_log.exists() else None
        observations.append({'original_prefix':old,'command_record_at_recovery':old_record,
                             'original_log_is_closed_loop':bool(old_record and 'research/evaluate_closed_loop.py' in old_record['command']),
                             'separately_copied_record':copied_record,
                             'copied_record_is_closed_loop':bool(copied_record and 'research/evaluate_closed_loop.py' in copied_record['command'])})
    (destination/'protected_before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    (destination/'collision_observed.json').write_text(json.dumps(observations,indent=2),encoding='utf-8')
    comparisons=[]
    for name,model,old,method,seed in configs:
        checkpoint=base/model/'best.pt'
        output=destination/name
        run([sys.executable,'research/evaluate_closed_loop.py','--method',method,'--split','test',
             '--episodes','21','--max-steps','300','--factory','research.contact_learning:load_policy',
             '--checkpoint',str(checkpoint),'--device','cpu','--seed',str(seed),
             '--output-dir',str(output),'--videos','0'],base/'logs','closed_loop_verify_'+name)
        new_summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        old_summary=json.loads((base/old/'summary.json').read_text(encoding='utf-8'))
        assert new_summary['episodes']==old_summary['episodes']
        assert new_summary['checkpoint_sha256']==old_summary['checkpoint_sha256']
        assert new_summary['controller_sha256']==frozen['controller_sha256']
        assert new_summary['evaluator_sha256']==frozen['evaluator_sha256']
        per_episode=[]
        for episode in new_summary['episodes']:
            filename=f'episode_{episode:06d}.npz'
            a=np.load(base/old/filename,allow_pickle=False)
            b=np.load(output/filename,allow_pickle=False)
            assert a.files==b.files
            fields={}
            for key in a.files:
                assert a[key].shape==b[key].shape and a[key].dtype==b[key].dtype
                exact=bool(np.array_equal(a[key],b[key]))
                maximum=float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64)))) if a[key].size else 0.
                # Coverage derives from GEOS area summation; physical states,
                # action targets, pixels, contact counts and flows remain strict.
                accepted=exact or (key=='coverage' and maximum<=1e-12)
                fields[key]={'exact_equal':exact,'max_absolute_difference':maximum,'accepted':accepted}
            per_episode.append({'episode_index':episode,'fields':fields,
                                'all_fields_accepted':all(x['accepted'] for x in fields.values())})
        item={'run':name,'original_directory':str(base/old),'fresh_directory':str(output),
              'command_record':str(base/'logs'/('closed_loop_verify_'+name+'.command.json')),
              'returncode':json.loads((base/'logs'/('closed_loop_verify_'+name+'.command.json')).read_text(encoding='utf-8'))['returncode'],
              'successes_original':old_summary['successes'],'successes_fresh':new_summary['successes'],
              'all_21_episode_arrays_accepted':all(x['all_fields_accepted'] for x in per_episode),
              'episodes':per_episode}
        comparisons.append(item)
        (destination/f'{name}_array_comparison.json').write_text(json.dumps(item,indent=2),encoding='utf-8')
        print(json.dumps({'verified':name,'successes':new_summary['successes'],
                          'arrays_match':item['all_21_episode_arrays_accepted']}),flush=True)
    unchanged=all(sha(ROOT/path)==expected for path,expected in before.items())
    passed=unchanged and all(row['all_21_episode_arrays_accepted'] for row in comparisons)
    report={'schema_version':1,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'passed':passed,'defect':'Offline evaluation and closed-loop evaluation reused log prefixes. Some command/stdout/stderr evidence was overwritten; trajectories, summaries and checkpoints were not.',
            'repair':'New fresh processes ran all 21 test episodes for each of four learned checkpoints. Log prefixes closed_loop_verify_* are unique; runner now refuses existing log files. Original records are not rewritten.',
            'old_command_records_at_recovery':observations,'protected_original_file_count':len(before),
            'protected_before_manifest':str(destination/'protected_before.json'),
            'all_original_trajectory_summary_checkpoint_files_unchanged':unchanged,
            'frozen_controller_sha256':frozen['controller_sha256'],'frozen_evaluator_sha256':frozen['evaluator_sha256'],
            'comparison_policy':'Every NPZ array compared. Exact equality required except coverage permits absolute error <= 1e-12 for GEOS summation order. New wall-clock times are independent measurements, not substituted for historical timing.',
            'runs':[{'run':x['run'],'successes_original':x['successes_original'],'successes_fresh':x['successes_fresh'],
                     'command_record':x['command_record'],'returncode':x['returncode'],
                     'all_21_episode_arrays_accepted':x['all_21_episode_arrays_accepted'],
                     'comparison_file':str(destination/(x['run']+'_array_comparison.json'))} for x in comparisons]}
    (base/'log_collision_recovery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':passed,'protected_files':len(before),'original_files_unchanged':unchanged}),flush=True)
    if not passed:raise AssertionError('Recovery comparison failed; inspect preserved reports')


if __name__=='__main__':main()
