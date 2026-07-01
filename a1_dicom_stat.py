import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
import shutil
import warnings

import pydicom
import tomlkit
from pydicom.errors import InvalidDicomError
from tqdm import tqdm

specific_tags = (
    'ImageType', 'Modality',
    'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
    'StudyDate', 'StudyTime',
    'Manufacturer', 'ManufacturerModelName',
)

specific_modalities = ('CT', 'PX')


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
        ds = pydicom.dcmread(file, stop_before_pixels=True, specific_tags=list(specific_tags))
    except InvalidDicomError:
        return

    tags = [ds.get(_) for _ in specific_tags]

    if None in tags:
        return

    if not ('ORIGINAL' in str(ds.ImageType) and 'PRIMARY' in str(ds.ImageType)):
        return

    if str(ds.Modality) not in specific_modalities:
        return

    return [str(_) for _ in tags]


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml', type=str)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

    dataset_root = Path(cfg['dataset']['root'])
    dataset_raw = dataset_root / 'raw'

    # 读取 DICOM 原始信息
    save_file_tags = dataset_root / 'raw_file_tags'
    if save_file_tags.exists():
        file_tags = pickle.loads(save_file_tags.read_bytes())
    else:
        files = [_.as_posix() for _ in dataset_raw.rglob('*') if _.is_file()]
        archives = [archive_folder(_) for _ in files]
        archives = [_ for _ in archives if _[1] is not None]

        # 解压压缩包
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

        files = [_.as_posix() for _ in dataset_raw.rglob('*') if _.is_file()]
        file_tags = {}

        # 多线程加速机械硬盘读写
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(dcmread, _): _ for _ in files}

            try:
                for fu in tqdm(as_completed(futures), 'Parsing DICOM', total=len(futures)):
                    try:
                        res = fu.result()
                        if res is not None:
                            file_tags[futures[fu]] = res
                    except Exception as _:
                        warnings.warn(f'{_} {futures[fu]}', stacklevel=2)

            except KeyboardInterrupt:
                print('Keyboard interrupted terminating...')
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                raise SystemExit

        save_file_tags.write_bytes(pickle.dumps(file_tags))

    # 统计摘要
    summary = []

    # 模态/设备型号分类统计
    for modality in specific_modalities:
        for uid_key in ('StudyInstanceUID', 'SeriesInstanceUID'):
            uid_k = uid_key.removesuffix('InstanceUID')

            tree = {}
            for file, tags in tqdm(file_tags.items(), f'{modality} {uid_k}'):
                tags = dict(zip(specific_tags, tags))

                if modality != tags['Modality']:
                    continue

                uid = tags[uid_key]
                mfm = ' '.join([tags['Manufacturer'], tags['ManufacturerModelName']])

                if mfm not in tree:
                    tree[mfm] = {}

                if uid not in tree[mfm]:
                    tree[mfm][uid] = {
                        'patient_id': tags['PatientID'],
                        'study_uid': tags['StudyInstanceUID'],
                        'study_date': tags['StudyDate'],
                        'study_time': tags['StudyTime'],
                        'files': [],
                    }

                tree[mfm][uid]['files'].append(Path(file).relative_to(dataset_raw).as_posix())

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
    for file, tags in tqdm(file_tags.items(), 'Patients'):
        tags = dict(zip(specific_tags, tags))

        patient_id = tags['PatientID']
        uid = tags['SeriesInstanceUID']
        modality = tags['Modality']
        mfm = ' '.join([tags['Manufacturer'], tags['ManufacturerModelName']])

        patients.add(patient_id)

        if mfm not in cfg['manufacturer_model'][modality]:
            continue

        if patient_id not in tree:
            tree[patient_id] = {_: {} for _ in specific_modalities}

        if uid not in tree[patient_id][modality]:
            tree[patient_id][modality][uid] = {
                'study_uid': tags['StudyInstanceUID'],
                'study_date': tags['StudyDate'],
                'study_time': tags['StudyTime'],
                'files': [],
            }

        tree[patient_id][modality][uid]['files'].append(Path(file).relative_to(dataset_raw).as_posix())

    single = {_: 0 for _ in specific_modalities}
    multiple = 0
    for patient_id in tree:
        is_single = False
        for modality in specific_modalities:
            if len(tree[patient_id][modality]) == sum(len(series) for series in tree[patient_id].values()):
                single[modality] += 1
                is_single = True
        if not is_single:
            multiple += 1

    summary.append('\n'.join([
        '```mermaid', f'pie title {len(tree)} Patients on modality',
        *[f'    "{_} only: {single[_]}": {single[_]}' for _ in specific_modalities],
        f'    "Multi-modality: {multiple}": {multiple}',
        '```',
    ]))

    # 统计摘要
    save_file_tags = Path(__file__).resolve().parent / 'SUMMARY.md'
    save_file_tags.write_text('\n\n'.join(summary), encoding='utf-8')


if __name__ == '__main__':
    launch()
