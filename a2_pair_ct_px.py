import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path
import shutil
import warnings
import tempfile

import tomlkit
from tqdm import tqdm
import numpy as np
from PIL import Image

from a1_raw_dicom import meta_tags


def dicom_float(v):
    if v is None:
        return None

    if not isinstance(v, (str, bytes)):
        try:
            if len(v) == 0:
                return None
            v = v[0]
        except TypeError:
            pass

    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def series_read(dataset_raw: Path, dataset_pair: Path, it: dict):
    patient_id = it['patient_id']
    category = it['category']
    series_uid = it['series_uid']
    files = [dataset_raw / _ for _ in it['files'].values()]

    import itk
    itk.ProcessObject.SetGlobalWarningDisplay(False)
    itk.OutputWindow.SetGlobalWarningDisplay(False)

    series_dir = dataset_pair / patient_id / category / series_uid
    series_dir.mkdir(parents=True, exist_ok=True)

    if len(files) == 0:
        return {**it, 'error': {'category': category, 'files': len(files)}}

    if category == 'PANORAMA':
        if len(files) > 1:
            return {**it, 'error': {'category': category, 'files': len(files)}}
        else:
            # read
            image = itk.imread(files[0].as_posix())
            origin = [float(_) for _ in itk.origin(image)]
            spacing = [float(_) for _ in itk.spacing(image)]
            size = [int(_) for _ in itk.size(image)]
            minmax = [float(_) for _ in itk.range(image)]

            if not (len(size) == 2 or (len(size) == 3 and size[2] == 1)):
                return {**it, 'error': {'category': category, 'dimension': len(size), 'size': size}}

            # snapshot
            a = itk.array_from_image(image)

            if a.dtype != np.uint8:
                w = dicom_float(it.get('meta', {}).get('WindowWidth')), dicom_float(it.get('meta', {}).get('WindowCenter'))

                if w[0] is None or w[1] is None or w[0] <= 0:
                    window = minmax
                else:
                    window = w[1] - w[0] * 0.5, w[1] + w[0] * 0.5

                a = np.clip((a - window[0]) * 255 / (window[1] - window[0]) + 0.5, 0, 255).astype(np.uint8)
            else:
                window = minmax

            # save
            if len(a.shape) == 2:
                a = a[np.newaxis, ...]
            for k in range(len(a)):
                Image.fromarray(np.ascontiguousarray(a[k].transpose(1, 0)), mode='L').save(series_dir / f'preview_{k}.png')
            itk.imwrite(image, series_dir / 'image.nii.gz')
            return {**it, 'meta': {'origin': origin, 'spacing': spacing, 'size': size, 'range': minmax, 'window': window}}
    elif category == 'CT':
        # sort
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
                return {**it, 'error': {'category': category, 'series_uids': series_uids}}

            file_names = names_generator.GetFileNames(series_uids[0])

            ImageType = itk.Image[itk.SS, 3]
            reader = itk.ImageSeriesReader[ImageType].New()
            reader.SetFileNames(file_names)
            reader.Update()

        # read
        image = reader.GetOutput()
        origin = [float(_) for _ in itk.origin(image)]
        spacing = [float(_) for _ in itk.spacing(image)]
        size = [int(_) for _ in itk.size(image)]
        minmax = [float(_) for _ in itk.range(image)]

        if True in [_ > 1.5 for _ in spacing]:
            return {**it, 'error': {'category': category, 'spacing': spacing}}

        # snapshot
        array = np.ascontiguousarray(itk.array_from_image(image).transpose(2, 1, 0))

        w = dicom_float(it.get('meta', {}).get('WindowWidth')), dicom_float(it.get('meta', {}).get('WindowCenter'))

        if w[0] is None or w[1] is None or w[0] <= 0:
            window = (0, 900)
        else:
            window = w[1] - w[0] * 0.5, w[1] + w[0] * 0.5

        previews = []
        for ax in range(3):
            a = array.copy()
            c = window[0] < a
            a *= c
            a = a.sum(axis=ax)
            c = np.sum(c, axis=ax)
            c[np.where(c <= 0)] = 1
            a = a / c

            a = (a - window[0]) * 255 / (window[1] - window[0]) + 0.5
            a = np.clip(a, 0, 255).astype(np.uint8)

            previews.append(a)

        # 非均匀采样，检查是否有文件缺失
        itk_meta = image.GetMetaDataDictionary()

        if itk_meta.HasKey('ITK_non_uniform_sampling_deviation'):
            d = float(itk_meta['ITK_non_uniform_sampling_deviation'])
            z = float(itk.spacing(image)[2])

            if z == 0 or d / z > 0.1:
                return {**it, 'error': {'category': category, 'non_uniform_sampling_deviation': d, 'slice_thickness': z}}

        # save
        for ax, a in enumerate(previews):
            Image.fromarray(a, mode='L').save(series_dir / f'preview_{ax}.png')
        itk.imwrite(image, series_dir / 'image.nii.gz')
        return {**it, 'meta': {'origin': origin, 'spacing': spacing, 'size': size, 'range': minmax, 'window': window}}
    else:
        return {**it, 'error': {'category': category}}


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.toml', type=str)
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

    dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
    dataset_raw = dataset_root / 'raw'
    dataset_pair = dataset_root / 'pair'

    save_raw_meta = dataset_root / 'raw_meta'
    if not save_raw_meta.exists():
        raise SystemError(f'Not found {save_raw_meta}')

    raw_meta = pickle.loads(save_raw_meta.read_bytes())

    # 重组序列
    series_metas, series_files = {}, {}
    for file, meta in tqdm(raw_meta.items(), 'Sort series'):
        if meta is None:
            continue

        meta = dict(zip(meta_tags, meta))

        patient_id = meta['PatientID']
        series_uid = meta['SeriesInstanceUID']
        sop_uid = meta['SOPInstanceUID']
        image_type = meta['ImageType']
        modality = meta['Modality']

        if modality != 'SC':
            if 'ORIGINAL' not in image_type or 'PRIMARY' not in image_type or 'LOCALIZER' in image_type:
                continue

        if patient_id not in series_metas:
            series_metas[patient_id] = {}

        if series_uid not in series_metas[patient_id]:
            series_metas[patient_id][series_uid] = meta

        if series_uid not in series_files:
            series_files[series_uid] = {}

        series_files[series_uid][sop_uid] = file

    # 筛选有效的配对数据
    pair_meta = {}
    only_ct, only_panorama = 0, 0
    for patient_id in tqdm(series_metas, 'Sort pairs'):
        ct_series = {_: series_metas[patient_id][_] for _ in series_metas[patient_id] if series_metas[patient_id][_]['Modality'] == 'CT'}

        # CT 只允许原始图像
        ct_series = {_: ct_series[_] for _ in ct_series}

        # PX 只允许原始图像
        px_series = {_: series_metas[patient_id][_] for _ in series_metas[patient_id] if series_metas[patient_id][_]['Modality'] == 'PX'}
        px_series = {_: px_series[_] for _ in px_series}

        # SC 只允许二次后处理的全景片图像
        sc_series = {_: series_metas[patient_id][_] for _ in series_metas[patient_id] if series_metas[patient_id][_]['Modality'] == 'SC'}
        sc_series = {_: sc_series[_] for _ in sc_series
                     if isinstance(sc_series[_]['ImageComments'], str) and 'PANORAMA' in sc_series[_]['ImageComments']}

        if len(ct_series) > 0 and len(px_series) == 0 and len(sc_series) == 0:
            only_ct += 1
            continue

        if len(ct_series) == 0 and len(px_series) + len(sc_series) > 0:
            only_panorama += 1
            continue

        if len(ct_series) > 0 and len(px_series) + len(sc_series) > 0:
            pair_meta[patient_id] = {'CT': ct_series, 'PANORAMA': {**px_series, **sc_series}}

    both_ct_panorama = len(pair_meta)
    print(f'Found {both_ct_panorama} patients with both CT and PANORAMA')
    print(f'Found {only_ct} patients with only CT')
    print(f'Found {only_panorama} patients with only PANORAMA')

    # 读取缓存
    save_pair_series = dataset_root / 'pair_series'
    if save_pair_series.exists():
        print(f'Loading cache from {save_pair_series}')
        pair_series = pickle.loads(save_pair_series.read_bytes())
    else:
        pair_series = {}

    # 增量文件
    new_series = {}
    for patient_id in pair_meta:
        for category in ('CT', 'PANORAMA'):
            for series_uid in pair_meta[patient_id][category]:
                it = pair_series.get(series_uid, {})

                if set(it.get('files', {}).keys()) == set(series_files[series_uid].keys()):
                    if 'meta' in it:
                        pair_meta[patient_id][category][series_uid].update(it['meta'])
                    continue

                new_series[series_uid] = {
                    'patient_id': patient_id,
                    'category': category,
                    'series_uid': series_uid,
                    'meta': pair_meta[patient_id][category][series_uid],
                    'files': series_files[series_uid],
                }

    print(f'Found {len(new_series)} new series')

    # 多线程加速机械硬盘读写
    if len(new_series):
        with ProcessPoolExecutor(max_workers=args.max_workers, max_tasks_per_child=1) as executor:
            futures = {executor.submit(series_read, dataset_raw, dataset_pair, it): it for _, it in new_series.items()}

            try:
                for fu in tqdm(as_completed(futures), 'Read series', len(futures)):
                    try:
                        it = fu.result()

                        patient_id = it['patient_id']
                        category = it['category']
                        series_uid = it['series_uid']

                        if 'meta' in it:
                            pair_meta[patient_id][category][series_uid].update(it['meta'])

                    except Exception as e:
                        it = futures[fu]
                        it = {**it, 'error': {'category': it['category'], 'exception': str(e)}}

                        patient_id = it['patient_id']
                        category = it['category']
                        series_uid = it['series_uid']

                        warnings.warn(f'{patient_id} {category} {series_uid} {e}')

                    pair_series[series_uid] = it

            except KeyboardInterrupt:
                print('Keyboard interrupted terminating...')
                executor.shutdown(wait=False)
                for future in futures:
                    future.cancel()
                raise SystemExit

        save_pair_series.parent.mkdir(parents=True, exist_ok=True)
        save_pair_series.write_bytes(pickle.dumps(pair_series))

    # 移除失败序列
    rm_patients = []
    for patient_id in pair_meta:
        rm_modalities = []

        for category in ('CT', 'PANORAMA'):
            rm_series = []

            for series_uid in pair_meta[patient_id][category]:
                if 'error' in pair_series.get(series_uid, 'error'):
                    rm_series.append(series_uid)

            for series_uid in rm_series:
                del pair_meta[patient_id][category][series_uid]

            if len(pair_meta[patient_id][category]) == 0:
                rm_modalities.append(category)

        for category in rm_modalities:
            del pair_meta[patient_id][category]

        if set(pair_meta[patient_id]) != {'CT', 'PANORAMA'}:
            rm_patients.append(patient_id)

    for patient_id in rm_patients:
        del pair_meta[patient_id]

    # 保存配对数据
    f = dataset_root / 'pair_meta'
    f.write_bytes(pickle.dumps(pair_meta))

    # 统计摘要
    summary = []
    summary.append('\n'.join([
        '```mermaid', f'pie title Patients on category',
        f'    "Pair CT/PANORAMA: {both_ct_panorama}" : {both_ct_panorama}',
        f'    "Only CT: {only_ct}" : {only_ct}',
        f'    "Only PANORAMA: {only_panorama}" : {only_panorama}',
        '```',
    ]))

    num = {}
    for patient_id in pair_meta:
        for category in ('CT', 'PANORAMA'):
            for series_uid in pair_meta[patient_id][category]:
                meta = pair_meta[patient_id][category][series_uid]
                modality = meta['Modality']

                if modality not in num:
                    num[modality] = 0

                num[modality] += 1

    summary.append(f'{len(pair_meta)} valid pairs on CT-PANORAMA')

    summary.append('\n'.join([
        '```mermaid', f'pie title Series on modality',
        *(f'    "{k}: {v}" : {v}' for k, v in num.items()),
        '```',
    ]))
    f = Path(__file__).parent / (Path(__file__).name.removesuffix('.py') + '.md')
    f.write_text('\n\n'.join(summary), encoding='utf-8')


if __name__ == '__main__':
    launch()
