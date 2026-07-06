import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
import shutil
import warnings

import pydicom
import pydicom.config
import tomlkit
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

from a0_define import major_tags, minor_tags, pair_modalities

pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE


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
    try:
        ds = pydicom.dcmread(file, stop_before_pixels=True, specific_tags=list(major_tags + minor_tags))
    except InvalidDicomError:
        return

    meta = [ds.get(_) for _ in major_tags]

    if None in meta:
        return

    meta += [ds.get(_) for _ in minor_tags]

    if not ('ORIGINAL' in str(ds.ImageType) and 'PRIMARY' in str(ds.ImageType)):
        return

    if str(ds.Modality) not in pair_modalities:
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
    files = [_.as_posix() for _ in dataset_raw.rglob('*') if _.is_file()]
    archives = [archive_folder(_) for _ in files]
    archives = [_ for _ in archives if _[1] is not None]

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
        except Exception as e:
            failed += 1
            warnings.warn(f'Unpacking error: {file} {e}', stacklevel=2)
            shutil.rmtree(folder, ignore_errors=True)

    print(f'Unpacking archives: {succeeded} succeeded {failed} failed')

    # 读取缓存
    save_raw_dicom_meta = dataset_root / 'raw_dicom_meta'
    if save_raw_dicom_meta.exists():
        print(f'Loading cache from {save_raw_dicom_meta}')
        dicom_meta = pickle.loads(save_raw_dicom_meta.read_bytes())
    else:
        dicom_meta = {}

    # 增量文件
    files = [_.relative_to(dataset_raw).as_posix() for _ in dataset_raw.rglob('*') if _.is_file()]

    new_files = [_ for _ in files if _ not in dicom_meta]
    print(f'Found {len(new_files)} new files in total {len(files)} files')

    # 多线程加速机械硬盘读写
    if len(new_files):
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(dcmread, dataset_raw / _): _ for _ in new_files}

            try:
                for fu in tqdm(as_completed(futures), 'Parsing DICOM', len(futures)):
                    try:
                        res = fu.result()
                        dicom_meta[futures[fu]] = res
                    except Exception as e:
                        warnings.warn(f'{e} {futures[fu]}', stacklevel=2)
                        dicom_meta[futures[fu]] = None

            except KeyboardInterrupt:
                print('Keyboard interrupted terminating...')
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                raise SystemExit

        save_raw_dicom_meta.write_bytes(pickle.dumps(dicom_meta))

    # 统计摘要
    summary = []

    # 模态/设备型号分类统计
    for modality in pair_modalities:
        for uid_key in ('StudyInstanceUID', 'SeriesInstanceUID'):
            uid_k = uid_key.removesuffix('InstanceUID')

            tree = {}
            for file, meta in tqdm(dicom_meta.items(), f'{modality} {uid_k}'):
                if meta is None:
                    continue

                meta = dict(zip(major_tags + minor_tags, meta))

                if modality != meta['Modality']:
                    continue

                uid = meta[uid_key]
                mfm = ' '.join([meta['Manufacturer'], meta['ManufacturerModelName']])

                if mfm not in tree:
                    tree[mfm] = set()

                if uid not in tree[mfm]:
                    tree[mfm].add(uid)

            nums = [(mfm, len(tree[mfm])) for mfm in tree]
            nums = sorted(nums, key=lambda x: x[1], reverse=True)

            summary.append('\n'.join([
                '```mermaid', f'pie title {sum(num for _, num in nums)} {modality} {uid_k} on manufacturer model',
                *[f'    "{('*' + mfm) if mfm in cfg['manufacturer_model'][modality] else mfm}: {num}" : {num}' for mfm, num in nums],
                '```',
            ]))

            summary.append('\n > \\*: specified manufacturer model')

    # 指定设备型号的患者/模态分类统计
    tree, patients = {}, set()
    for file, meta in tqdm(dicom_meta.items(), 'Patients'):
        if meta is None:
            continue

        meta = dict(zip(major_tags + minor_tags, meta))

        patient_id = meta['PatientID']
        uid = meta['SeriesInstanceUID']
        modality = meta['Modality']
        mfm = ' '.join([meta['Manufacturer'], meta['ManufacturerModelName']])

        patients.add(patient_id)

        if mfm not in cfg['manufacturer_model'][modality]:
            continue

        if patient_id not in tree:
            tree[patient_id] = {}

        if modality not in tree[patient_id]:
            tree[patient_id][modality] = set()

        if uid not in tree[patient_id][modality]:
            tree[patient_id][modality].add(uid)

    single = {_: 0 for _ in pair_modalities}
    pair = 0
    for patient_id in tree:
        is_single = False
        for modality in pair_modalities:
            if modality in tree[patient_id] and len(tree[patient_id].keys()) == 1:
                single[modality] += 1
                is_single = True
        if not is_single:
            pair += 1

    summary.append('\n'.join([
        '```mermaid', f'pie title {len(tree)} Patients on modality',
        *[f'    "{_} only: {single[_]}": {single[_]}' for _ in pair_modalities],
        f'    "Pair: {pair}": {pair}',
        '```',
    ]))

    # 统计摘要
    f = Path(__file__).resolve().parent / 'SUMMARY.md'
    f.write_text('\n\n'.join(summary), encoding='utf-8')


if __name__ == '__main__':
    launch()
