# tmux new -d -s a4 'uv run streamlit run a4_align_ct_px.py --server.port 8501 2>&1 | tee a4.log'

import argparse
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import streamlit as st
import tomlkit
import cv2
import tomlkit
from PIL import Image


def resize_for_spacing(img, row_spacing, col_spacing):
    h, w = img.shape[:2]
    new_h = max(1, round(h * row_spacing / col_spacing))
    return cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)


st.set_page_config('Nonapx', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### CT/PANORAMA 对齐')

if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='config.toml', type=str)
        args = parser.parse_args()

        cfg = tomlkit.loads(Path(args.config).read_text('utf-8')).unwrap()

        dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
        dataset_raw = dataset_root / 'raw'
        dataset_pair = dataset_root / 'pair'

        save_pair_meta = dataset_root / 'pair_meta'
        if not save_pair_meta.exists():
            st.error(f'配对数据不存在 {save_pair_meta.as_posix()}')
            st.stop()

        pair_meta: dict = pickle.loads(save_pair_meta.read_bytes())

        if len(pair_meta) == 0:
            st.error(f'配对数据为空 {save_pair_meta.as_posix()}')
            st.stop()

    st.session_state['init'] = cfg, pair_meta
    st.rerun()

elif (it := st.session_state.get('select')) is None:
    cfg, pair_meta = st.session_state['init']

    dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
    dataset_pair = dataset_root / 'pair'

    st.metric('Valid pairs', len(pair_meta))

    def fn_patient(_):
        return f'{_} CT {len(pair_meta[_]['CT'])} PANORAMA {len(pair_meta[_]['PANORAMA'])}'

    patient_id = st.selectbox('Patient ID', list(sorted(pair_meta.keys())), format_func=fn_patient)

    if patient_id is None:
        st.stop()

    def fn_series(category, _):
        meta = pair_meta[patient_id][category][_]
        for _ in meta:
            if meta[_] is None:
                meta[_] = ''

        mfm = ' '.join([_ for _ in (meta['Manufacturer'], meta['ManufacturerModelName']) if _ != ''])
        text = [str(meta['Modality']), mfm]

        if category == 'CT':
            text.append(str(meta['StudyDescription']))
            text.append(str(meta['SeriesDescription']))
            text.append(str(meta['size']))
            text.append(str(meta['spacing']))
            text.append(str(meta['range']))
            text.append(str(meta['window']))
        elif category == 'PANORAMA':
            text.append(str(meta['StudyDescription']))
            text.append(str(meta['SeriesDescription']))
            text.append(str(meta['size'][:2]))
            text.append(str(meta['spacing'][:2]))
            text.append(str(meta['range']))
            text.append(str(meta['window']))
        return ' '.join(text)

    ct_series_id = st.radio('CT Series UID', list(sorted(pair_meta[patient_id]['CT'])), format_func=partial(fn_series, 'CT'))

    if ct_series_id is None:
        st.stop()

    if st.button('导出 CT'):
        with st.spinner('正在打包'):
            f = dataset_pair / patient_id / 'CT' / ct_series_id / f'image.nii.gz'
            st.download_button('下载', data=f.read_bytes(), file_name=f'{patient_id}_CT_{ct_series_id}.nii.gz')

    meta = pair_meta[patient_id]['CT'][ct_series_id]
    for _ in meta:
        if meta[_] is None:
            meta[_] = ''

    previews = []
    for i in range(10):
        f = dataset_pair / patient_id / 'CT' / ct_series_id / f'preview_{i}.png'
        if not f.exists():
            break
        previews.append(np.flipud(np.array(Image.open(f.as_posix())).transpose(1, 0)))

    for ax, col in enumerate(st.columns(3, vertical_alignment='bottom')):
        ax = 2 - ax
        bc = [_ for _ in range(3) if _ != ax]
        col.image(resize_for_spacing(previews[ax], meta['spacing'][bc[1]], meta['spacing'][bc[0]]), ['侧位', '正位', '轴位'][ax])

    st.code(tomlkit.dumps(meta, True), 'toml')

    pa_series_id = st.radio('PANORAMA Series UID', list(sorted(pair_meta[patient_id]['PANORAMA'])), format_func=partial(fn_series, 'PANORAMA'))

    if pa_series_id is None:
        st.stop()

    if st.button('导出 PANORAMA'):
        with st.spinner('正在打包'):
            f = dataset_pair / patient_id / 'PANORAMA' / pa_series_id / f'image.nii.gz'
            st.download_button('下载', data=f.read_bytes(), file_name=f'{patient_id}_PANORAMA_{pa_series_id}.nii.gz')

    meta = pair_meta[patient_id]['PANORAMA'][pa_series_id]
    for _ in meta:
        if meta[_] is None:
            meta[_] = ''

    previews = []
    for i in range(10):
        f = dataset_pair / patient_id / 'PANORAMA' / pa_series_id / f'preview_{i}.png'
        if not f.exists():
            break
        previews.append(np.array(Image.open(f.as_posix())).transpose(1, 0))

    for i in range(len(previews)):
        st.image(previews[i], f'第 {i + 1} 帧')

    st.code(tomlkit.dumps(meta, True), 'toml')
