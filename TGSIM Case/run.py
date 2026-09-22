"""Fresh, budget-matched TGSIM training, selection and evaluation pipeline."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

CASE = Path(__file__).resolve().parent
ROOT = CASE.parent
sys.path[:0] = [str(CASE), str(ROOT)]
TRAINERS = {
    'residual_marl': ('RL.train_ppo', ['--residual-mode', 'candidate_logits']),
    'residual_param': ('RL.train_ppo', ['--residual-mode', 'param_delta']),
    'direct_discrete_rl': ('Baselines.train_direct_discrete_rl', ['--minibatch-size', '512']),
    'mappo': ('Baselines.train_marl', ['--algo', 'mappo']),
}
STEMS = dict(residual_marl='residual_policy', residual_param='residual_param_policy',
             direct_discrete_rl='direct_discrete_policy', mappo='mappo_policy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'train', 'newbaselines', 'bench', 'all'])
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run = (args.run_dir or CASE/'runs'/('full_'+stamp)).resolve()
    os.environ['TGSIM_RUN_DIR'] = str(run)
    import config as c
    from RL.experiment_protocol import source_manifest, write_json
    c.LOG_DIR.mkdir(parents=True, exist_ok=True)
    c.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol='tgsim_recorded_initialization_v5_calibration_dynamics', recipe='paper_2day',
        created_utc=datetime.now(timezone.utc).isoformat(),
        run_dir=str(run), seeds=c.SEEDS, update_limit=c.TRAIN_UPDATES,
        validation_episodes=c.VAL_EPISODES, validation_every_updates=c.VAL_EVERY,
        test_scenarios=c.BENCH_SCENARIOS, agents=c.NUM_AGENTS, max_steps=c.MAX_STEPS,
        offline_optimizer_steps=c.NEW_BASELINE_STEPS,
        calibration=str(c.CALIBRATION), curb=str(c.STREET_BOUNDARIES),
        vehicle_length_m=c.VEHICLE_LENGTH, vehicle_width_m=c.VEHICLE_WIDTH,
        vehicle_wheelbase_m=c.VEHICLE_WHEELBASE, max_agent_speed_mps=c.MAX_AGENT_SPEED,
        source_hashes=source_manifest(),
        trajectory_sha256=hashlib.sha256(c.TRAJECTORIES_CSV.read_bytes()).hexdigest())
    path = run/'run_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ('protocol', 'recipe', 'agents', 'max_steps', 'update_limit'):
            if previous.get(key) != manifest.get(key):
                raise ValueError('Run settings changed (%s); use a fresh run directory.' % key)
        if previous.get('trajectory_sha256') != manifest['trajectory_sha256']:
            raise ValueError('TGSIM prepared trajectories changed; use a fresh run directory and retrain.')
        site_inputs = ('Calibration/utility_calibration_tgsim.json',
                       'data/TGSIM FB/derived_boundaries/street_boundaries.csv')
        old_hashes = previous.get('source_hashes', {})
        new_hashes = manifest['source_hashes']
        changed_site_inputs = [name for name in site_inputs
                               if old_hashes.get(name) != new_hashes.get(name)]
        if changed_site_inputs:
            raise ValueError('TGSIM calibration or curb changed since this run: '
                             + ', '.join(changed_site_inputs)
                             + '. Use a fresh run directory and retrain before benchmarking.')
        if previous['source_hashes'] != manifest['source_hashes'] and not args.resume:
            raise ValueError('Code changed since this run. Use a fresh run directory.')
    else:
        write_json(path, manifest)
    state_path = run/'status.json'
    if state_path.exists() and not args.resume:
        raise ValueError('Run already exists; use --resume to skip completed stages')
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(jobs={})
    state.update(status='running', pid=os.getpid(), command=args.command, started_utc=datetime.now(timezone.utc).isoformat())
    write_json(state_path, state)
    write_json(CASE/'runs/latest_run.json', dict(run_dir=str(run), pid=os.getpid(), status_file=str(state_path)))
    lock = threading.Lock()

    def launch(key, command, gpu=None):
        with lock:
            if args.resume and state['jobs'].get(key, {}).get('status') == 'complete':
                print('SKIP completed '+key, flush=True)
                return
        logfile = c.LOG_DIR/(key+'.log')
        env = os.environ.copy()
        env['TGSIM_TORCH_THREADS'] = '1'
        if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        cmd = [sys.executable, '-u', str(CASE/'run_module.py'), *map(str, command)]
        start = time.time()
        with logfile.open('w', encoding='utf-8') as handle:
            process = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            with lock:
                state['jobs'][key] = dict(status='running', pid=process.pid, command=cmd, log=str(logfile), started=time.time())
                write_json(state_path, state)
                print('START '+key+' pid='+str(process.pid), flush=True)
            code = process.wait()
        with lock:
            state['jobs'][key].update(status='complete' if code == 0 else 'failed', exit_code=code,
                                     wall_time_s=time.time()-start)
            write_json(state_path, state)
            print(('OK ' if code == 0 else 'FAILED ')+key, flush=True)
        if code: raise RuntimeError(key+' failed; see '+str(logfile))

    def parallel(jobs, workers):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(launch, *job) for job in jobs]
            for future in as_completed(futures): future.result()

    try:
        if args.command in ('smoke', 'all'):
            launch('verify', ['verify'])
            launch('verify_learning', ['tools.verify_site_learning'])
        if args.command in ('train', 'all'):
            common = ['--updates', c.TRAIN_UPDATES, '--episodes-per-update', 4,
                '--num-agents', c.NUM_AGENTS, '--max-steps', c.MAX_STEPS,
                '--collision-penalty', 0, '--collision-event-penalty', 1, '--gamma', .95,
                '--val-every', c.VAL_EVERY, '--val-episodes', c.VAL_EPISODES,
                '--test-episodes', c.TEST_EPISODES,
                '--validation-seed-start', 910000, '--test-seed-start', 810000,
                '--calibration', c.CALIBRATION, '--skip-test', '--log-every', 1]
            jobs = []
            for name in c.TRAIN_MODELS:
                for seed in c.SEEDS:
                    module, extra = TRAINERS[name]
                    save = c.CHECKPOINT_DIR/(STEMS[name]+'_seed'+str(seed)+'.pt')
                    jobs.append((name+'_seed'+str(seed), [module, *extra, *common, '--seed', seed, '--save', save]))
            parallel(jobs, max(1, args.jobs))
            for name in c.TRAIN_MODELS:
                for seed in c.SEEDS:
                    report = json.loads((c.CHECKPOINT_DIR/(STEMS[name]+'_seed'+str(seed)+'.summary.json')).read_text())
                    completed = int(report.get('updates') or (report.get('budget') or {}).get('policy_updates') or 0)
                    if completed < c.TRAIN_UPDATES:
                        raise ValueError(name+' seed '+str(seed)+' did not reach the 2-day update budget')
        if args.command in ('newbaselines', 'all'):
            launch('prepare_offline', ['prepare_new_baselines'])
            import torch
            devices = max(1, torch.cuda.device_count())
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # One worker per physical GPU. Each model/seed is trained then selected
            # before that GPU is reused. No duplicate evaluation of offline exports.
            bundles = [(name, seed) for seed in c.SEEDS for name in c.NEW_BASELINE_MODELS]
            def generation_worker(gpu):
                for name, seed in bundles[gpu::devices]:
                    key = name+'_seed'+str(seed)
                    raw = c.CHECKPOINT_DIR/(name+'_raw_seed'+str(seed)+'.pt')
                    launch(key+'_train', ['new_baselines_cli', '--threads', 1, 'train', name,
                        '--data', c.NEW_BASELINE_DATA, '--steps', c.NEW_BASELINE_STEPS,
                        '--batch-size', 4, '--eval-every', c.NEW_BASELINE_STEPS//5,
                        '--seed', seed, '--device', device, '--output', raw], gpu)
                    launch(key+'_select', ['new_baselines_cli', '--threads', 1, 'select', name,
                        '--candidates', raw.with_suffix('.candidates'), '--device', device,
                        '--output', c.CHECKPOINT_DIR/(name+'_policy_seed'+str(seed)+'.pt'),
                        '--num-agents', c.NUM_AGENTS, '--max-steps', c.MAX_STEPS,
                        '--val-episodes', c.VAL_EPISODES, '--validation-seed-start', 910000,
                        '--test-episodes', c.TEST_EPISODES, '--test-seed-start', 810000,
                        '--calibration', c.CALIBRATION], gpu)
            with ThreadPoolExecutor(max_workers=devices) as pool:
                for future in as_completed([pool.submit(generation_worker, gpu) for gpu in range(devices)]):
                    future.result()
        if args.command in ('bench', 'all'):
            launch('benchmark', ['Baselines.benchmark', '--models', *c.BENCH_MODELS,
                '--seed', 810000, '--scenarios', c.BENCH_SCENARIOS, '--num-agents', c.NUM_AGENTS,
                '--max-steps', c.MAX_STEPS, '--calibration', c.CALIBRATION,
                '--checkpoint-dir', c.CHECKPOINT_DIR,
                '--residual-checkpoint', c.CHECKPOINT_DIR/'residual_policy.pt',
                '--train-seeds', *c.SEEDS, '--require-matched-protocol', '--n-boot', c.N_BOOT,
                '--output-dir', c.RESULT_DIR, '--reference-model', 'residual_marl',
                '--conflict-lookahead', 'full', '--no-figures'])
            launch('param_stress', ['Baselines.ablation_stress', '--mode', 'both',
                '--lookahead', 'both', '--models', *c.PARAM_EVAL_MODELS,
                '--scenarios', c.BENCH_SCENARIOS, '--stress-scenarios', c.BENCH_SCENARIOS,
                '--num-agents', c.NUM_AGENTS, '--stress-agents', c.STRESS_AGENTS,
                '--max-steps', c.MAX_STEPS, '--calibration', c.CALIBRATION,
                '--checkpoint-dir', c.CHECKPOINT_DIR,
                '--residual-checkpoint', c.CHECKPOINT_DIR/'residual_policy.pt',
                '--train-seeds', *c.SEEDS, '--n-boot', c.N_BOOT,
                '--output-dir', c.RESULT_DIR/'param_lookahead_stress', '--no-figures'])
            launch('gate_ablation', ['Baselines.ablation_stress', '--mode', 'gate',
                '--lookahead', 'full', '--scenarios', c.BENCH_SCENARIOS,
                '--num-agents', c.NUM_AGENTS, '--max-steps', c.MAX_STEPS,
                '--calibration', c.CALIBRATION, '--checkpoint-dir', c.CHECKPOINT_DIR,
                '--residual-checkpoint', c.CHECKPOINT_DIR/'residual_policy.pt',
                '--train-seeds', *c.SEEDS, '--n-boot', c.N_BOOT,
                '--output-dir', c.RESULT_DIR/'gate_ablation', '--no-figures'])
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)


if __name__ == '__main__':
    main()
