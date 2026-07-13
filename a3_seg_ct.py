# uv run python -O a3_seg_ct.py
# tmux new -d -s a3 'uv run python -O a3_seg_ct.py 2>&1 | tee a3.log'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path
import warnings

import tomlkit
from tqdm import tqdm


def totalseg(task: str, input_file: str | Path, output_file: str | Path):
    input_file = Path(input_file)
    output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(input_file.as_posix(), output_file.as_posix(), ml=True, task=task, quiet=True,
                     nr_thr_resamp=1, nr_thr_saving=1, force_split=(task == 'teeth'), device='gpu')


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml', type=str)
    parser.add_argument('--max_workers', type=int, default=1)
    parser.add_argument('--warmup', action='store_true', default=False)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

    dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
    dataset_pair = dataset_root / 'pair'

    save_pair_meta = dataset_root / 'pair_meta'
    if not save_pair_meta.exists():
        raise SystemError(f'Not found {save_pair_meta}')

    pair_meta = pickle.loads(save_pair_meta.read_bytes())

    save_errors = dataset_root / 'totalsegmentator_errors'
    if save_errors.exists():
        errors = pickle.loads(save_errors.read_bytes())
    else:
        errors = {}

    # 统计增量任务
    totalseg_tasks = {
        'total': {},
        'teeth': {},
        'craniofacial_structures': {},
        'head_glands_cavities': {},
        'head_muscles': {},
        'headneck_bones_vessels': {},
    }
    for patient_id in pair_meta:
        for category in ('CT', ):
            for series_uid in pair_meta[patient_id][category]:
                for task in totalseg_tasks:
                    input_file = dataset_pair / patient_id / category / series_uid / 'image.nii.gz'
                    output_file = dataset_root / 'totalsegmentator' / series_uid / f'{task}.nii.gz'

                    if input_file.exists() and not output_file.exists() and series_uid not in errors.get(task, {}):
                        totalseg_tasks[task][series_uid] = (task, input_file.as_posix(), output_file.as_posix())

    print('TotalSegmentator', ' '.join([f'{task} {len(totalseg_tasks[task])}' for task in totalseg_tasks]))

    # 预热下载模型
    if args.warmup:
        for task in totalseg_tasks:
            print(f'TotalSegmentator {task}: warm up')
            if len(totalseg_tasks[task]):
                items = list(totalseg_tasks[task].items())
                totalseg(*items[0][1])
    elif args.max_workers < 2:
        for task in totalseg_tasks:
            if len(totalseg_tasks[task]) == 0:
                continue

            for series_uid, _ in tqdm(totalseg_tasks[task].items(), f'TotalSegmentator {task}', len(totalseg_tasks[task])):
                try:
                    totalseg(*_)
                except Exception as e:
                    if _[0] not in errors:
                        errors[_[0]] = {}
                    errors[_[0]][series_uid] = str(e)
                    save_errors.write_bytes(pickle.dumps(errors))
                    warnings.warn(f'TotalSegmentator {_}: {e}')
    else:
        for task in totalseg_tasks:
            if len(totalseg_tasks[task]) == 0:
                continue

            with ProcessPoolExecutor(max_workers=args.max_workers, max_tasks_per_child=1) as executor:
                futures = {executor.submit(totalseg, *_): (series_uid, _) for series_uid, _ in totalseg_tasks[task].items()}

                try:
                    for fu in tqdm(as_completed(futures), f'TotalSegmentator {task}', len(futures)):
                        try:
                            fu.result()
                        except Exception as e:
                            series_uid, _ = futures[fu]
                            if _[0] not in errors:
                                errors[_[0]] = {}
                            errors[_[0]][series_uid] = str(e)
                            save_errors.write_bytes(pickle.dumps(errors))
                            warnings.warn(f'TotalSegmentator {_}: {e}')

                except KeyboardInterrupt:
                    print('Keyboard interrupted terminating...')
                    executor.shutdown(wait=False)
                    for future in futures:
                        future.cancel()
                    raise SystemExit


if __name__ == '__main__':
    launch()
