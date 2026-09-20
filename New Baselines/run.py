"""One entry point: prepare, train, smoke. Evaluation uses Baselines.benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
for path in (ROOT, PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from new_baselines.config import Config
from new_baselines.data import prepare
from new_baselines.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare', help='Prepare disjoint recorded train/validation scenes')
    p.add_argument('--csv', type=Path, default=ROOT/'data/Lebanon_Highway/Final_Lebanon_Data.csv')
    p.add_argument('--output', type=Path, default=ROOT/'New Baselines/artifacts/data')
    p.add_argument('--config', type=Path)
    p.add_argument('--train-scenes', type=int, default=20)
    p.add_argument('--validation-scenes', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-id', type=int, default=2)
    p.add_argument('--lane-kf', type=int, default=1)
    p = subs.add_parser('train', help='Train on the local training split, select by validation loss')
    p.add_argument('model', choices=['ctrl_sim', 'ctg_plus_plus'])
    p.add_argument('--data', type=Path, default=ROOT/'New Baselines/artifacts/data')
    p.add_argument('--output', type=Path)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--config', type=Path, help='Optional config; data geometry/horizons must match cache')
    p.add_argument('--smoke-only', action='store_true', help='Mark checkpoint ineligible for paper benchmarks')
    p = subs.add_parser('select', help='Select saved candidates with the common closed-loop validation rule')
    p.add_argument('model', choices=['ctrl_sim', 'ctg_plus_plus'])
    p.add_argument('--candidates', type=Path, required=True, help='Directory of step_*.pt validation candidates')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val-episodes', type=int, default=16)
    p.add_argument('--validation-seed-start', type=int, default=910000)
    p.add_argument('--test-episodes', type=int, default=16)
    p.add_argument('--test-seed-start', type=int, default=810000)
    p.add_argument('--num-agents', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=240)
    p.add_argument('--run-id', type=int, default=2)
    p.add_argument('--lane-kf', type=int, default=1)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--dense-spawn', action='store_true')
    p = subs.add_parser('smoke', help='Prepare tiny real-data splits, train and roll out both models')
    p.add_argument('--csv', type=Path, default=ROOT/'data/Lebanon_Highway/Final_Lebanon_Data.csv')
    p.add_argument('--output', type=Path, default=ROOT/'New Baselines/artifacts/smoke')
    p.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == 'prepare':
        cfg = Config(**json.loads(args.config.read_text())) if args.config else Config()
        for split, count in [('train', args.train_scenes), ('validation', args.validation_scenes)]:
            manifest = prepare(args.csv, args.output/(split+'.npz'), cfg, count, split,
                               args.seed, args.run_id, args.lane_kf)
            print(f'{split}: {manifest["scene_count"]} recorded scenes', flush=True)
    elif args.command == 'train':
        config = json.loads(args.config.read_text()) if args.config else json.loads((args.data/'train.json').read_text())['config']
        cfg = Config(**config).validate()
        output = args.output or ROOT/'New Baselines/checkpoints'/(args.model+'_policy.pt')
        train(args.model, args.data/'train.npz', args.data/'validation.npz', output, cfg,
            steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
            device=args.device, smoke=args.smoke_only, eval_every=args.eval_every)
    elif args.command == 'select':
        from new_baselines.training import select_checkpoints
        select_checkpoints(args.model, sorted(args.candidates.glob('step_*.pt')), args.output, args, device=args.device)
    else:
        smoke(args)


def smoke(args):
    from Baselines.runner import rollout
    from Baselines.scenario import build_scenario
    from new_baselines.controller import TrafficGenerationController
    started = time.perf_counter()
    cfg = Config(hidden=32, layers=1, decoder_layers=1, max_agents=4, context=4,
                 history=3, horizon=4, diffusion_steps=4, return_bins=32)
    output = args.output
    for split in ('train', 'validation'):
        prepare(args.csv, output/(split+'.npz'), cfg, count=2, split=split)
        print('Prepared '+split, flush=True)
    results = {}
    for name in ('ctrl_sim', 'ctg_plus_plus'):
        checkpoint = output/(name+'_policy.pt')
        payload = train(name, output/'train.npz', output/'validation.npz', checkpoint, cfg,
                        steps=2, batch_size=2, device=args.device, smoke=True)
        controller = TrafficGenerationController(name, checkpoint, device=args.device, allow_smoke=True)
        scenario = build_scenario(seed=42, num_agents=3, max_steps=3, dt=cfg.dt)
        result = rollout(scenario, controller)
        results[name] = dict(training_steps=2, validation_loss=payload['metadata']['best_validation_loss'],
            rollout_steps=result.steps, collision_events=result.collision_events,
            offroad_steps=result.offroad_steps, wall_time=result.wall_time)
    summary = dict(smoke_only=True, note='Execution check only; not a performance comparison.',
                   elapsed_s=time.perf_counter()-started, models=results)
    (output/'smoke_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
