import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
import shutil
import warnings

import tomlkit
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

required_tags = (
    'ImageType', 'Modality',
    'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
)

meta_tags = required_tags + (
    'StudyDate', 'StudyTime',
    'PatientName', 'PatientSex', 'PatientAge',
    'Manufacturer', 'ManufacturerModelName',
    'NumberOfFrames',
    'StudyDescription', 'SeriesDescription',
    'WindowCenter', 'WindowWidth',
    'ImageOrientationPatient',
    'ImageComments',
)


def archive_folder(file: str | Path):
    file = Path(file)
    for suffix in (
        '.zip',
        '.tar', '.tar.gz', '.tar.bz2', '.tar.xz',
        '.tgz', '.tbz2', '.txz',
    ):
        if file.name.lower().endswith(suffix):
            return file, file.parent / file.name[:-len(suffix)]
    return file, None


def dcmread(file: str | Path):
    import pydicom
    import pydicom.config

    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

    try:
        ds = pydicom.dcmread(file, stop_before_pixels=True, specific_tags=list(meta_tags))
    except InvalidDicomError:
        return

    meta = [ds.get(_) for _ in meta_tags]
    meta = [_ if _ != '' else None for _ in meta]

    if None in meta[:len(required_tags)]:
        return

    return [str(_) if _ is not None else None for _ in meta]


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml', type=str)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

    dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
    dataset_raw = dataset_root / 'raw'

    # 解压压缩包
    archives = [archive_folder(_) for _ in tqdm(dataset_raw.rglob('*'), 'Scan archives') if _.is_file()]
    archives = [_ for _ in tqdm(archives, 'Scan archives') if _[1] is not None]
    print(f'Found {len(archives)} archives')

    if len(archives):
        succeeded, failed = 0, 0
        for file, folder in tqdm(archives, 'Unpacking archives'):
            if folder.exists():
                shutil.rmtree(folder)

            folder.mkdir()

            try:
                if file.name.lower().endswith('.zip'):
                    shutil.unpack_archive(file, folder)
                else:
                    shutil.unpack_archive(file, folder, filter='data')
                succeeded += 1
                file.unlink()
            except Exception as e:
                failed += 1
                warnings.warn(f'Unpacking error: {file} {e}', stacklevel=2)
                shutil.rmtree(folder, ignore_errors=True)

        print(f'Unpacking archives: {succeeded} succeeded {failed} failed')

    # 读取缓存
    save_raw_meta = dataset_root / 'raw_meta'
    if save_raw_meta.exists():
        print(f'Loading cache from {save_raw_meta}')
        raw_meta = pickle.loads(save_raw_meta.read_bytes())
    else:
        raw_meta = {}

    # 增量文件
    new_files = [_.relative_to(dataset_raw).as_posix() for _ in tqdm(dataset_raw.rglob('*'), 'Scan new files')
                 if _.is_file() and _.relative_to(dataset_raw).as_posix() not in raw_meta]
    print(f'Found {len(new_files)} new files')

    # 多线程加速机械硬盘读写
    if len(new_files):
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(dcmread, dataset_raw / _): _ for _ in new_files}

            try:
                for fu in tqdm(as_completed(futures), 'Parsing DICOM', len(futures)):
                    try:
                        raw_meta[futures[fu]] = fu.result()
                    except Exception as e:
                        warnings.warn(f'{e} {futures[fu]}', stacklevel=2)
                        raw_meta[futures[fu]] = None

            except KeyboardInterrupt:
                print('Keyboard interrupted terminating...')
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                raise SystemExit

        save_raw_meta.write_bytes(pickle.dumps(raw_meta))

    # 统计模态
    modalities = set()
    for file, meta in tqdm(raw_meta.items(), f'Scan modalities'):
        if meta is None:
            continue
        meta = dict(zip(meta_tags, meta))
        if meta['Modality'] is not None:
            modalities.add(meta['Modality'])
    print(f'Found {len(modalities)} modalities')

    # 统计每种模态的设备品牌型号
    summary = []
    for modality in modalities:
        for i, uid_key in enumerate(('PatientID', 'StudyInstanceUID', 'SeriesInstanceUID')):
            name = ['Patient', 'Study', 'Series'][i]

            tree = {}
            for file, meta in tqdm(raw_meta.items(), f'{modality} {name}'):
                if meta is None:
                    continue

                meta = dict(zip(meta_tags, meta))

                if modality != meta['Modality']:
                    continue

                mfm = ' '.join([_ for _ in (meta['Manufacturer'], meta['ManufacturerModelName']) if _ is not None and _ != ''])

                if mfm not in tree:
                    tree[mfm] = set()

                uid = meta[uid_key]

                if uid not in tree[mfm]:
                    tree[mfm].add(uid)

            nums = [(mfm, len(tree[mfm])) for mfm in tree]
            nums = sorted(nums, key=lambda x: x[1], reverse=True)

            summary.append('\n'.join([
                '```mermaid', f'pie title {sum(num for _, num in nums)} {modality} {name} on manufacturer model',
                *[f'    "{mfm}: {num}" : {num}' for mfm, num in nums],
                '```',
            ]))

    # 统计摘要
    f = Path(__file__).parent / (Path(__file__).name.removesuffix('.py') + '.md')
    f.write_text('\n\n'.join(summary), encoding='utf-8')


if __name__ == '__main__':
    launch()
