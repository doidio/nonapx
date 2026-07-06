import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import pickle
from pathlib import Path
import shutil
import warnings
import tempfile

import tomlkit
from tqdm import tqdm

from a0_define import major_tags, minor_tags, pair_modalities


def main(files: list[Path], series_dir: Path):
    import itk
    itk.ProcessObject.SetGlobalWarningDisplay(False)
    itk.OutputWindow.SetGlobalWarningDisplay(False)

    series_dir.mkdir(parents=True, exist_ok=True)
    modality = series_dir.parent.name

    if len(files) == 0:
        return {'modality': modality, 'files': len(files)}
    elif modality == 'PX':
        if len(files) > 1:
            return {'modality': modality, 'files': len(files)}
        else:
            image = itk.imread(files[0].as_posix(), itk.SS)
            size = [int(_) for _ in itk.size(image)]

            if len(size) == 2:
                import numpy as np
                array = itk.array_from_image(image)
                array_3d = np.expand_dims(array, axis=0)
                image_3d = itk.image_from_array(array_3d)

                spacing = itk.spacing(image)
                image_3d.SetSpacing([spacing[0], spacing[1], 1.0])

                origin = itk.origin(image)
                image_3d.SetOrigin([origin[0], origin[1], 0.0])

                image = image_3d
                size = [int(_) for _ in itk.size(image)]

            if len(size) == 3:
                itk.imwrite(image, series_dir / 'image.nii.gz')
            else:
                return {'modality': modality, 'size': size}
    elif modality == 'CT':
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i, f in enumerate(files):
                filename = f'{i}_{f.name}'
                try:
                    (tmpdir / filename).symlink_to(f)
                except Exception:
                    shutil.copy(f, tmpdir / filename)

            names_generator = itk.GDCMSeriesFileNames.New()
            names_generator.SetDirectory(tmpdir.as_posix())
            series_uids = names_generator.GetSeriesUIDs()

            if len(series_uids) != 1:
                return {'series_uids': series_uids}

            file_names = names_generator.GetFileNames(series_uids[0])

            ImageType = itk.Image[itk.SS, 3]
            reader = itk.ImageSeriesReader[ImageType].New()
            reader.SetFileNames(file_names)
            reader.Update()

        image = reader.GetOutput()

        meta_dict = dict(image)

        # 非均匀采样，检查是否有文件缺失
        if 'ITK_non_uniform_sampling_deviation' in meta_dict:
            deviation = float(meta_dict['ITK_non_uniform_sampling_deviation'])
            slice_thickness = float(itk.spacing(image)[2])
            if deviation / slice_thickness < 0.1:
                itk.imwrite(image, series_dir / 'image.nii.gz')
            return {'non_uniform_sampling_deviation': deviation, 'slice_thickness': slice_thickness}
        else:
            itk.imwrite(image, series_dir / 'image.nii.gz')
    else:
        return {'modality': modality, 'files': len(files)}


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml', type=str)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

    dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
    dataset_raw = dataset_root / 'raw'
    dataset_pair = dataset_root / 'pair'

    save_raw_dicom_meta = dataset_root / 'raw_dicom_meta'
    if not save_raw_dicom_meta.exists():
        raise SystemError(f'Not found {save_raw_dicom_meta}')

    dicom_meta = pickle.loads(save_raw_dicom_meta.read_bytes())

    tree, patients = {}, set()
    for file, meta in tqdm(dicom_meta.items(), 'Patients'):
        if meta is None:
            continue

        meta = dict(zip(major_tags + minor_tags, meta))

        patient_id = meta['PatientID']
        series_uid = meta['SeriesInstanceUID']
        modality = meta['Modality']
        mfm = ' '.join([meta['Manufacturer'], meta['ManufacturerModelName']])

        patients.add(patient_id)

        if mfm not in cfg['manufacturer_model'][modality]:
            continue

        if patient_id not in tree:
            tree[patient_id] = {}

        if modality not in tree[patient_id]:
            tree[patient_id][modality] = {}

        if series_uid not in tree[patient_id][modality]:
            tree[patient_id][modality][series_uid] = {**meta, 'files': []}

        tree[patient_id][modality][series_uid]['files'].append(file)

    # 筛选全模态数据
    pair_modality_patients = set()
    for patient_id in tree:
        if False not in [_ in tree[patient_id] for _ in pair_modalities]:
            pair_modality_patients.add(patient_id)

    print(f'Pair modality patients: {len(pair_modality_patients)}')

    # 读取缓存
    save_series_errors = dataset_root / 'series_errors'
    if save_series_errors.exists():
        print(f'Loading cache from {save_series_errors}')
        series_errors = pickle.loads(save_series_errors.read_bytes())
    else:
        series_errors = {}

    # 增量文件
    threads, total = [], 0
    for patient_id in pair_modality_patients:
        for modality in tree[patient_id]:
            for series_uid in tree[patient_id][modality]:
                total += 1

                if series_uid in series_errors:
                    continue

                series_dir = dataset_pair / patient_id / modality / series_uid
                files = [dataset_raw / _ for _ in tree[patient_id][modality][series_uid]['files']]
                threads.append([series_uid, files, series_dir])

    print(f'Found {len(threads)} new series in total {total} series')

    # 多线程加速机械硬盘读写
    if len(threads):
        with ProcessPoolExecutor(max_workers=args.max_workers, max_tasks_per_child=1) as executor:
            futures = {executor.submit(main, *_[1:]): _[0] for _ in threads}

            try:
                for fu in tqdm(as_completed(futures), 'DICOM Series to NIFTI', len(futures)):
                    try:
                        series_errors[futures[fu]] = fu.result()
                    except Exception as e:
                        warnings.warn(f'{e} {futures[fu]}')
                        series_errors[futures[fu]] = {'exception': str(e)}

            except KeyboardInterrupt:
                print('Keyboard interrupted terminating...')
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                raise SystemExit

        save_series_errors.parent.mkdir(parents=True, exist_ok=True)
        save_series_errors.write_bytes(pickle.dumps(series_errors))

    # 保存元信息
    pair_tree = {_: deepcopy(tree[_]) for _ in pair_modality_patients}
    for patient_id in pair_tree:
        for modality in pair_tree[patient_id]:
            for series_uid in pair_tree[patient_id][modality]:
                del pair_tree[patient_id][modality][series_uid]['files']

    f = dataset_root / 'pair_meta'
    f.write_bytes(pickle.dumps(pair_tree))


if __name__ == '__main__':
    launch()
