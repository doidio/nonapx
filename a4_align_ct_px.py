# tmux new -d -s a4 'xvfb-run -a uv run streamlit run a4_align_ct_px.py --server.port 8501 --server.headless true 2>&1 | tee a4.log'

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
import itk
import pyvista as pv

from a4_kernel import extract_surface_wp


def resize_for_spacing(img, row_spacing, col_spacing):
    h, w = img.shape[:2]
    new_h = max(1, round(h * row_spacing / col_spacing))
    return cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)


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
        text.append(str(tuple(meta['size'])))
        text.append(str(tuple(meta['spacing'])))
        text.append(str(tuple(meta['range'])))
        text.append(str(tuple(meta['window'])))
    elif category == 'PANORAMA':
        text.append(str(meta['StudyDescription']))
        text.append(str(meta['SeriesDescription']))
        text.append(str(tuple(meta['size'][:2])))
        text.append(str(tuple(meta['spacing'][:2])))
        text.append(str(tuple(meta['range'])))
        text.append(str(tuple(meta['window'])))
    return ' '.join([_ for _ in text if len(_)])


st.set_page_config('Nonapx', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### CT/PANORAMA 筛选 配对 对齐')

if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='config.toml', type=str)
        parser.add_argument('--start', default=0, type=int)
        parser.add_argument('--num', default=100, type=int)
        parser.add_argument('--user', default='Who', type=str)
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
        total = len(pair_meta)

        keys = list(sorted(pair_meta.keys()))[args.start:args.start + args.num]
        pair_meta = {_: pair_meta[_] for _ in keys}

        if len(pair_meta) == 0:
            st.error(f'配对数据为空 {save_pair_meta.as_posix()}')
            st.stop()

    st.session_state['init'] = cfg, pair_meta, (args.start, args.num, total, args.user)
    st.rerun()

elif (it := st.session_state.get('selected')) is None:
    cfg, pair_meta, (start, num, total, user) = st.session_state['init']

    dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
    dataset_pair = dataset_root / 'pair'

    st.metric(f'分工 {start + 1} - {start + len(pair_meta)}', f'{user}', f'子进度 {0} / {len(pair_meta)}',
              delta_arrow='off', delta_description=f'总进度 {0} / {total}')

    cols = st.columns((5, 2))

    with cols[0]:
        st.subheader('筛选')

    with cols[0]:
        def fn_patient(_):
            return ' '.join([str(_), 'CT', str(len(pair_meta[_]['CT'])), 'PANORAMA', str(len(pair_meta[_]['PANORAMA']))])

        patient_id = st.selectbox('**选择 Patient ID**', list(sorted(pair_meta.keys())), format_func=fn_patient, width=500)

        if patient_id is None:
            st.stop()

        ct_series_uid = st.radio('**选择 CT Series**', list(sorted(pair_meta[patient_id]['CT'])), format_func=partial(fn_series, 'CT'))

        if ct_series_uid is None:
            st.stop()

        with st.expander(ct_series_uid, False):
            st.code(tomlkit.dumps(pair_meta[patient_id]['CT'][ct_series_uid], True), 'toml')

        # if st.button('导出 CT'):
        #     with st.spinner('正在打包'):
        #         f = dataset_pair / patient_id / 'CT' / ct_series_uid / f'image.nii.gz'
        #         st.download_button('下载', data=f.read_bytes(), file_name=f'{patient_id}_CT_{ct_series_uid}.nii.gz')

        meta = pair_meta[patient_id]['CT'][ct_series_uid]
        for _ in meta:
            if meta[_] is None:
                meta[_] = ''

        previews = []
        for i in range(10):
            f = dataset_pair / patient_id / 'CT' / ct_series_uid / f'preview_{i}.png'
            if not f.exists():
                break
            _ = np.array(Image.open(f.as_posix())).transpose(1, 0)
            if i in (0, 1):
                _ = np.flipud(_)
            previews.append(_)

        for ax, col in enumerate(st.columns(3, vertical_alignment='bottom')):
            ax = 2 - ax
            bc = [_ for _ in range(3) if _ != ax]
            col.image(resize_for_spacing(previews[ax], meta['spacing'][bc[1]], meta['spacing'][bc[0]]), ['侧位', '正位', '轴位'][ax])

        pano_series_uid = st.radio('**选择 PANORAMA Series**',
                                   list(sorted(pair_meta[patient_id]['PANORAMA'])), format_func=partial(fn_series, 'PANORAMA'))

        if pano_series_uid is None:
            st.stop()

        with st.expander(pano_series_uid, False):
            st.code(tomlkit.dumps(pair_meta[patient_id]['PANORAMA'][pano_series_uid], True), 'toml')

        # if st.button('导出 PANORAMA'):
        #     with st.spinner('正在打包'):
        #         f = dataset_pair / patient_id / 'PANORAMA' / pa_series_id / f'image.nii.gz'
        #         st.download_button('下载', data=f.read_bytes(), file_name=f'{patient_id}_PANORAMA_{pa_series_id}.nii.gz')

        meta = pair_meta[patient_id]['PANORAMA'][pano_series_uid]
        for _ in meta:
            if meta[_] is None:
                meta[_] = ''

        previews = []
        for i in range(10):
            f = dataset_pair / patient_id / 'PANORAMA' / pano_series_uid / f'preview_{i}.png'
            if not f.exists():
                break
            previews.append(np.array(Image.open(f.as_posix())).transpose(1, 0))

        for i in range(len(previews)):
            st.image(resize_for_spacing(previews[i], meta['spacing'][1], meta['spacing'][0]), f'第 {i + 1} 帧')

    with cols[1]:
        st.subheader('配对')

    subcols = cols[1].columns(2)

    if 'select' not in st.session_state or patient_id not in st.session_state['select']:
        st.session_state['select'] = {patient_id: {'CT': set(), 'PANORAMA': None}}

    with subcols[1]:
        if st.button('清空', key='clear_ct'):
            st.session_state['select'][patient_id]['CT'] = set()

    with subcols[0]:
        if st.button('CT 多选', width='stretch'):
            if ct_series_uid not in st.session_state['select'][patient_id]['CT']:
                st.session_state['select'][patient_id]['CT'].add(ct_series_uid)

    with cols[1]:
        warns = set()
        roi = {_: [set() for _ in range(3)] for _ in ('size', 'spacing', 'origin')}
        for ct_series_uid in sorted(st.session_state['select'][patient_id]['CT']):
            st.text(ct_series_uid)

            meta = pair_meta[patient_id]['CT'][ct_series_uid]
            for _ in ('size', 'spacing', 'origin'):
                warn = set()
                for i in range(3):
                    roi[_][i].add(meta[_][i])

                    if len(roi[_][i]) > 1:
                        warn.add('XYZ'[i])

                if len(warn):
                    warns.add(_ + ' ' + ''.join(sorted(warn)))

        if len(warns):
            warns = ' '.join(sorted(warns))
            st.warning(f'{warns} 不一致，检查 CT 同源性')

    subcols = cols[1].columns(2)

    with subcols[1]:
        if st.button('清空', key='clear_panorama'):
            st.session_state['select'][patient_id]['PANORAMA'] = None

    with subcols[0]:
        if st.button('PANORAMA 单选', width='stretch'):
            st.session_state['select'][patient_id]['PANORAMA'] = pano_series_uid

    with cols[1]:
        if (pano_series_uid := st.session_state['select'][patient_id]['PANORAMA']):
            st.text(pano_series_uid)

    subcols = cols[1].columns(2)

    with subcols[0]:
        ct = st.session_state['select'][patient_id]['CT']
        panorama = st.session_state['select'][patient_id]['PANORAMA']
        if len(ct) and panorama and st.button('下一步', width='stretch'):
            st.session_state['selected'] = st.session_state['select']
            st.rerun()

else:
    cfg, pair_meta, (start, num, total, user) = st.session_state['init']

    dataset_root = Path(cfg['dataset']['root']).resolve().absolute()
    dataset_pair = dataset_root / 'pair'

    st.metric(f'分工 {start + 1} - {start + len(pair_meta)}', f'{user}', f'子进度 {0} / {len(pair_meta)}',
              delta_arrow='off', delta_description=f'总进度 {0} / {total}')

    if (_ := len(st.session_state['select'])) != 1:
        st.error(f'非同一患者 {_}')
        st.stop()

    patient_id = list(st.session_state['select'].keys())[0]

    st.caption(f'患者 {patient_id}')

    ct_series_uids = st.session_state['select'][patient_id]['CT']
    pano_series_uid = st.session_state['select'][patient_id]['PANORAMA']

    for ct_series_uid in ct_series_uids:
        with st.expander(fn_series('CT', ct_series_uid), False):
            st.code(tomlkit.dumps(pair_meta[patient_id]['CT'][ct_series_uid], True), 'toml')

    with st.expander(fn_series('PANORAMA', pano_series_uid), False):
        st.code(tomlkit.dumps(pair_meta[patient_id]['PANORAMA'][pano_series_uid], True), 'toml')

    # 载入 CT 和全景片
    if 'images' not in st.session_state:
        with st.spinner('载入图像'):
            images = {}

            for series_uid in [pano_series_uid, *ct_series_uids]:
                category = 'PANORAMA' if series_uid == pano_series_uid else 'CT'

                f = dataset_pair / patient_id / category / series_uid / 'image.nii.gz'
                image = itk.imread(f.as_posix())
                origin = [float(_) for _ in itk.origin(image)]
                spacing = [float(_) for _ in itk.spacing(image)]
                size = [int(_) for _ in itk.size(image)]
                minmax = [int(_) for _ in itk.range(image)]

                a = itk.array_from_image(image)

                if series_uid == pano_series_uid:
                    if len(a.shape) == 3 and a.shape[0] == 1:
                        a = a[0]
                        origin = origin[1:]
                        spacing = spacing[1:]
                        size = size[1:]

                    if len(a.shape) != 2:
                        st.error(f'不支持的 {category} shape={a.shape}')
                        st.stop()

                a = np.ascontiguousarray(a.transpose(*[_ for _ in reversed(range(len(a.shape)))]))
                images[series_uid] = (a, origin, spacing, size, minmax)

            st.session_state['images'] = images

    images = st.session_state['images']

    cols = st.columns((2, 1))
    views = [_.container() for _ in cols]

    # 全景片
    if 'panorama' not in st.session_state:
        a, origin, spacing, size, minmax = images[pano_series_uid]
        meta = pair_meta[patient_id]['PANORAMA'][pano_series_uid]

        if a.dtype != np.uint8:
            w = meta['window']
            a = np.clip((a - w[0]) * 255 / (w[1] - w[0]) + 0.5, 0, 255).astype(np.uint8)

        a = resize_for_spacing(a.transpose(1, 0), spacing[1], spacing[0])
        st.session_state['panorama'] = a

    panorama = st.session_state['panorama']
    with cols[0]:
        views[0].image(panorama, 'PANORAMA')

    cols_3d = cols[1].columns((panorama.shape[1], panorama.shape[0]))

    with cols_3d[0]:
        view_type: str = st.radio('视角', ['右侧位', '右斜位', '正位', '左斜位', '左侧位', '轴位'], 2, horizontal=True)

    with cols_3d[1]:
        minmax = min([images[_][-1][0] for _ in ct_series_uids]), max([images[_][-1][1] for _ in ct_series_uids])
        bone_min = st.number_input(f'骨阈值 [{minmax[0]}, {minmax[1]}]', *minmax, 250, 10)

    # 面网格重建
    if 'bone_3d' not in st.session_state or st.session_state['bone_3d'][0] != bone_min:
        bone_meshes = {}
        with st.spinner('三维重建'):
            for series_uid in ct_series_uids:
                a, origin, spacing, size, _ = images[series_uid]
                mesh = extract_surface_wp(a, origin, spacing, bone_min)
                bone_meshes[series_uid] = mesh

        st.session_state['bone_3d'] = bone_min, bone_meshes

    bone_min, bone_meshes = st.session_state['bone_3d']

    # 模拟射线源

    # 3D 可视化
    pl = pv.Plotter(off_screen=True, border=False, window_size=[panorama.shape[0], panorama.shape[0]],
                    line_smoothing=True, point_smoothing=True, polygon_smoothing=True)
    pl.enable_parallel_projection()
    pl.enable_depth_peeling()
    pl.enable_anti_aliasing('msaa')

    bounds = None
    colors = list(pv.hex_colors.keys())[:len(bone_meshes)]
    for i, series_uid in enumerate(bone_meshes):
        mesh = bone_meshes[series_uid]
        pl.add_mesh(mesh, color=colors[i], render=False)

        if bounds is None:
            bounds = np.array(mesh.bounds)
        else:
            bounds = np.array([np.min([bounds[0], mesh.bounds[0]], axis=0), np.max([bounds[1], mesh.bounds[1]], axis=0)])

    assert bounds is not None

    if view_type == '右侧位':
        pl.camera_position = 'xz'
        pl.camera.Azimuth(-90)
    elif view_type == '右斜位':
        pl.camera_position = 'xz'
        pl.camera.Azimuth(-45)
    elif view_type == '正位':
        pl.camera_position = 'xz'
    elif view_type == '左斜位':
        pl.camera_position = 'xz'
        pl.camera.Azimuth(45)
    elif view_type == '左侧位':
        pl.camera_position = 'xz'
        pl.camera.Azimuth(90)
    elif view_type == '轴位':
        pl.camera_position = 'xy'
        pl.camera.Elevation(180)

    b = bounds - np.array([0, 0.5*(bounds[1][1] - bounds[0][1]), 0])
    pl.reset_camera(bounds=b.T.flatten())
    pl.camera.Zoom(1.5)
    # pl.camera.parallel_scale = (bounds[1][2] - bounds[0][2]) * 0.5
    # pl.reset_camera_clipping_range()
    # pl.render()

    with cols[1]:
        views[1].image(np.array(pl.screenshot(return_img=True)), 'CT')

    pl.close()
    del pl
