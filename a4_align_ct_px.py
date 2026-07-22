# tmux new -d -s a4 'xvfb-run -a uv run streamlit run a4_align_ct_px.py --server.port 8501 --server.headless true 2>&1 | tee a4.log'

import argparse
import pickle
from pathlib import Path
from functools import partial
import re
from zipfile import ZipFile
import io
import warp as wp

import numpy as np
import streamlit as st
import tomlkit
import cv2
import tomlkit
from PIL import Image
import itk
import pyvista as pv

from a3_seg_ct import totalseg_task_names
from a4_kernel import extract_surface_wp


def resize_for_spacing(img, row_spacing, col_spacing):
    h, w = img.shape[:2]
    new_h = max(1, round(h * row_spacing / col_spacing))
    return cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)


def parse_spacing_from_comments(comments_str):
    """
    从 ImageComments 中解析出 spacing (mm) 和 放大率 MAGFACT
    """
    if not comments_str:
        return (None, None), 1.0

    # 1. 匹配 Pixel-XSize 和 Pixel-YSize，支持小数以及 um/mm 单位
    match_x = re.search(r'Pixel-XSize:([\d.]+)\s*(um|mm)?', comments_str)
    match_y = re.search(r'Pixel-YSize:([\d.]+)\s*(um|mm)?', comments_str)

    # 2. 匹配放大因子 MAGFACT
    match_mag = re.search(r'MAGFACT:([\d.]+)', comments_str)
    mag_fact = float(match_mag.group(1)) if match_mag else 1.0
    spacing_x = None
    spacing_y = None
    if match_x:
        val_x, unit_x = float(match_x.group(1)), match_x.group(2) or 'um'
        spacing_x = val_x / 1000.0 if unit_x == 'um' else val_x
    if match_y:
        val_y, unit_y = float(match_y.group(1)), match_y.group(2) or 'um'
        spacing_y = val_y / 1000.0 if unit_y == 'um' else val_y
    return (spacing_x, spacing_y), mag_fact


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
st.markdown('### CT/PANORAMA 模拟对齐')

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

        if st.button('导出 CT'):
            with st.spinner('正在打包'):
                files = [dataset_pair / patient_id / 'CT' / ct_series_uid / f'image.nii.gz']
                for name in totalseg_task_names:
                    file = dataset_root / 'totalsegmentator' / ct_series_uid / f'{name}.nii.gz'
                    if file.exists():
                        files.append(file)

                buffer = io.BytesIO()
                with ZipFile(buffer, 'w') as zipf:
                    for file in files:
                        zipf.write(file, arcname=file.name)
                buffer.seek(0)

                st.download_button('下载', data=buffer.read(), file_name=f'{patient_id}_CT_{ct_series_uid}.zip')

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
                        origin = origin[:2]
                        spacing = spacing[:2]
                        size = size[:2]

                    if len(a.shape) != 2:
                        st.error(f'不支持的 {category} shape={a.shape}')
                        st.stop()

                a = np.ascontiguousarray(a.transpose(*[_ for _ in reversed(range(len(a.shape)))]))
                images[series_uid] = (a, origin, spacing, size, minmax)

            st.session_state['images'] = images

    images = st.session_state['images']

    # 全景片
    if len([_ for _ in ('panorama', 'detector_spacing', 'detector_magfact') if _ not in st.session_state]):
        a, origin, spacing, size, minmax = images[pano_series_uid]
        meta = pair_meta[patient_id]['PANORAMA'][pano_series_uid]

        if a.dtype != np.uint8:
            w = meta['window']
            a = np.clip((a - w[0]) * 255 / (w[1] - w[0]) + 0.5, 0, 255).astype(np.uint8)

        a = resize_for_spacing(a.transpose(1, 0), spacing[1], spacing[0])
        st.session_state['panorama'] = a

        # 探测器像素间距
        st.session_state['detector_spacing'] = 0.1

        if meta['Modality'] == 'PX':  # Planmeca ProMax 0.08 mm Sirona ORTHOPHOS XG 0.108003 mm
            st.session_state['detector_spacing'] = (spacing[0] + spacing[1]) * 0.5
            st.session_state['detector_magfact'] = 1.1
        elif meta['Modality'] == 'SC':  # 0.144 mm
            spacing, magfact = parse_spacing_from_comments(meta['ImageComments'])
            if spacing[0] is not None and spacing[1] is not None:
                st.session_state['detector_spacing'] = (spacing[0] + spacing[1]) * 0.5
                st.session_state['detector_magfact'] = magfact

    panorama = st.session_state['panorama']
    assert panorama.shape[1] > 1
    pano_img = panorama.copy()

    _ = st.columns((panorama.shape[1], panorama.shape[0]), vertical_alignment='bottom')
    view_3d_head = _[1].columns((3, 1))
    views = [_.container() for _ in _]

    _ = st.columns((panorama.shape[1], panorama.shape[0]))
    view_2d_foot = _[0].columns(5)
    view_3d_foot = _[1].columns(5)

    detector_spacing = st.session_state['detector_spacing']
    detector_magfact = st.session_state['detector_magfact']
    st.write(detector_spacing)

    # 面网格重建
    with view_3d_head[1]:
        minmax = min([images[_][-1][0] for _ in ct_series_uids]), max([images[_][-1][1] for _ in ct_series_uids])
        bone_min = st.number_input(f'骨阈值 [{minmax[0]}, {minmax[1]}]', *minmax, 250, 10, key='bone_min')

    if len([_ for _ in ('bone_3d', 'bone_bounds', 'bone_center', 'bone_hl') if _ not in st.session_state]) or st.session_state['bone_3d'][0] != bone_min:
        bone_meshes, bone_bounds = {}, None
        with st.spinner('三维重建'):
            for series_uid in ct_series_uids:
                a, origin, spacing, size, _ = images[series_uid]
                mesh = extract_surface_wp(a, origin, spacing, bone_min)
                bone_meshes[series_uid] = mesh

                if bone_bounds is None:
                    bone_bounds = np.array(mesh.bounds)
                else:
                    bone_bounds = np.array([np.min([bone_bounds[0], mesh.bounds[0]], axis=0), np.max([bone_bounds[1], mesh.bounds[1]], axis=0)])

        assert bone_bounds is not None

        bone_center = np.mean(bone_bounds, axis=0)
        bone_hl = np.ceil((bone_bounds[1] - bone_bounds[0]) / 0.2) * 0.1
        bone_bounds[0] = bone_center - bone_hl
        bone_bounds[1] = bone_center + bone_hl

        st.session_state['bone_3d'] = bone_min, bone_meshes
        st.session_state['bone_bounds'] = bone_bounds
        st.session_state['bone_center'] = bone_center
        st.session_state['bone_hl'] = bone_hl

    bone_min, bone_meshes = st.session_state['bone_3d']
    bone_bounds = st.session_state['bone_bounds']
    bone_center = st.session_state['bone_center']
    bone_hl = st.session_state['bone_hl']

    # 扫描坐标系，原点在弓形弧顶，左右对称
    with view_3d_foot[4]:
        if st.button('重置'):
            del st.session_state['scan_cs']

    if len([_ for _ in ('scan_cs', ) if _ not in st.session_state]):
        st.session_state['scan_cs'] = [
            bone_center[0], bone_bounds[0][1], bone_center[2],
            0.0, 0.0, 0.0,  # XYZ euler in degrees
        ]

    with view_3d_foot[3]:
        step = st.radio('步长 mm/deg', (0.1, 0.5, 1.0, 5, 10, 50, 100), 3)

    def move_scan(v_scan):
        q = wp.quat_from_euler(wp.vec3([wp.radians(float(_)) for _ in st.session_state['scan_cs'][3:]]), 1, 0, 2)
        R = np.asarray(wp.quat_to_matrix(q)).reshape(3, 3)
        v_ct = R @ np.asarray(v_scan, dtype=float)
        for i in range(3):
            st.session_state['scan_cs'][i] += float(v_ct[i])

    with view_3d_foot[0]:
        if st.button('前 Y-', width='stretch'):
            move_scan([0, -step, 0])
        if st.button('右 X-', width='stretch'):
            move_scan([-step, 0, 0])
        if st.button('横滚 Roll-', width='stretch'):
            st.session_state['scan_cs'][4] -= step
        if st.button('俯仰 Pitch-', width='stretch'):
            st.session_state['scan_cs'][3] -= step

    with view_3d_foot[1]:
        if st.button('上 Z+', width='stretch'):
            move_scan([0, 0, step])
        if st.button('下 Z-', width='stretch'):
            move_scan([0, 0, -step])
        if st.button('偏转 Yaw-', width='stretch'):
            st.session_state['scan_cs'][5] -= step
        if st.button('偏转 Yaw+', width='stretch'):
            st.session_state['scan_cs'][5] += step

    with view_3d_foot[2]:
        if st.button('后 Y+', width='stretch'):
            move_scan([0, step, 0])
        if st.button('左 X+', width='stretch'):
            move_scan([step, 0, 0])
        if st.button('横滚 Roll+', width='stretch'):
            st.session_state['scan_cs'][4] += step
        if st.button('俯仰 Pitch+', width='stretch'):
            st.session_state['scan_cs'][3] += step

    # 先旋转全景角 Z 再旋转俯仰角 X 最后旋转横滚角 Y，随动轴 ZXY 即固定轴 YXZ
    scan_cs = [_ for _ in st.session_state['scan_cs']]
    scan_cs_rot = wp.quat_from_euler(wp.vec3([wp.radians(float(_)) for _ in scan_cs[3:]]), 1, 0, 2)
    scan_cs_pos = np.array(scan_cs[:3])

    to_scan = np.eye(4)
    to_scan[:3, :3] = (R := np.asarray(wp.quat_to_matrix(scan_cs_rot)).reshape(3, 3).T)  # type: ignore
    to_scan[:3, 3] = -R @ scan_cs_pos

    from_scan_wp = wp.transform(wp.vec3(scan_cs_pos), scan_cs_rot)
    to_scan_wp = wp.transform_inverse(from_scan_wp)

    # 牙齿、骨皮质等在透视方向上较薄的结构特征，远源点小角度和近源点大角度可能生成结构相似的局部图像
    # 焦层中心曲面 focal-trough center surface，等距平行于射线源轨迹 X-ray source trajectory
    h = panorama.shape[0] * detector_spacing / detector_magfact / bone_hl[2] / 2
    scan_size_default = tuple(np.ceil(float(_ * h) / 0.1) * 0.1 for _ in bone_hl)

    if len([_ for _ in ('scan_size', ) if _ not in st.session_state]):
        st.session_state['scan_size'] = list(scan_size_default)

    scan_size = st.session_state['scan_size']

    with view_2d_foot[0]:
        for i in range(3):
            _ = ['左右半宽', '前后纵深', '上下半高'][i]
            scan_size[i] = st.number_input(f'{_} 0 ~ {scan_size_default[i] * 2:.1f} mm', 0.1,
                                           scan_size_default[i] * 2, scan_size[i], 0.1, '%.1f', key=f'scan_size_{i}')
    st.session_state['scan_size'] = scan_size

    # 弓形中点切线长度
    with view_2d_foot[1]:
        arch_level = st.number_input('弓形系数 0.1 ~ 1.9', 0.1, 1.9, 1.0, 0.1, '%.1f', key='arch_level')
        mid_tg = arch_level * scan_size[0]

    # 三次参数曲线
    def scan_spline(u):
        u = np.asarray(u)
        x = (scan_size[0] - mid_tg) * u ** 3 + mid_tg * u
        y = scan_size[1] * u ** 2
        return np.column_stack((x, y, np.zeros_like(u)))

    # 三次参数曲线斜率
    def scan_spline_derivative(u):
        u = np.asarray(u)
        x = 3 * (scan_size[0] - mid_tg) * u ** 2 + mid_tg
        y = 2 * scan_size[1] * u
        return np.column_stack((x, y, np.zeros_like(u)))

    # 近似弧长均匀
    u_dense = np.linspace(-1.0, 1.0, panorama.shape[1] * 4)
    curve_dense = scan_spline(u_dense)

    seg_len = np.linalg.norm(np.diff(curve_dense, axis=0), axis=1)
    arc_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc_len /= arc_len[-1]

    # 曲线采样
    u = np.interp(np.linspace(0.0, 1.0, panorama.shape[1]), arc_len, u_dense)
    scan_curve = scan_spline(u)

    # 焦层中心坐标系
    scan_axis_u = scan_spline_derivative(u)
    scan_axis_u /= np.linalg.norm(scan_axis_u, axis=1, keepdims=True)

    scan_axis_z = np.broadcast_to(np.array([0.0, 0.0, 1.0]), scan_axis_u.shape)
    scan_axis_y = np.cross(scan_axis_z, scan_axis_u)
    scan_axis_y /= np.linalg.norm(scan_axis_y, axis=1, keepdims=True)

    if len([_ for _ in ('scan_plane_height', ) if _ not in st.session_state]):
        st.session_state['scan_plane_height'] = panorama.shape[0] * detector_spacing / detector_magfact

    scan_plane_height = st.session_state['scan_plane_height']

    # 相机内参
    with view_2d_foot[2]:
        # 估计射线源到探测器距离 Source-Detector Distance
        sdd = (bone_bounds[1][1] - bone_bounds[0][1]) * 2.0

        # 估计相机焦距 focal length
        # unit = round(sdd / detector_spacing / 10)
        # focal_length = st.number_input(f'焦距 [{unit}, {unit * 20}] px', unit, unit * 20, unit * 10)

        # 主点
        # principal_point = np.array([slit_hw, (panorama.shape[0] - 1) / 2])
        # intrisics = np.array([
        #     [focal_length, 0, principal_point[0]],
        #     [0, focal_length, principal_point[1]],
        #     [0, 0, 1],
        # ])

    # 2D 可视化
    with views[0]:
        st.image(pano_img, f'全景 {panorama.shape[1]} x {panorama.shape[0]}')

    # 3D 可视化
    pl = pv.Plotter(off_screen=True, border=False, window_size=[panorama.shape[0], panorama.shape[0]],
                    line_smoothing=True, point_smoothing=True, polygon_smoothing=True)
    pl.enable_parallel_projection()
    # pl.enable_depth_peeling()  # xvfb 不支持半透明物体深度排序
    pl.enable_anti_aliasing('msaa')

    view_settings = {
        '↖': (30, -30), '↑': (0, -90), '↗': (-30, -30),
        '←': (90, 0), '+': (0, 0), '→': (-90, 0),
        '↙': (30, 30), '↓': (0, 90), '↘': (-30, 30),
    }

    with view_3d_head[0]:
        view_type: str = st.radio('视图', [*view_settings.keys(), '#'], horizontal=True)

    # 绘制颅骨
    actor_bounds = None
    colors = list(pv.hex_colors.keys())[:len(bone_meshes)]
    for i, series_uid in enumerate(bone_meshes):
        actor = pl.add_mesh(bone_meshes[series_uid], color=colors[i], render=False)
        actor.user_matrix = to_scan
        bounds = np.array(actor.bounds).reshape(3, 2).T

        if actor_bounds is None:
            actor_bounds = bounds
        else:
            actor_bounds = np.array([np.min([actor_bounds[_], bounds[_]], axis=0) for _ in range(2)])

    assert actor_bounds is not None

    actor_bounds -= np.mean(actor_bounds, axis=0)

    # 绘制焦层中心曲线
    axis_y, axis_z = [], []
    for i in range(0, len(scan_curve), max(1, len(scan_curve) // 100)):
        axis_y.append(scan_curve[i] - scan_axis_y[i] * 5)
        axis_y.append(scan_curve[i] + scan_axis_y[i] * 5)
        axis_z.append(scan_curve[i] - scan_axis_z[i] * scan_size[2])
        axis_z.append(scan_curve[i] + scan_axis_z[i] * scan_size[2])

    pl.add_lines(np.asarray(axis_y), connected=False, color=[0.25, 1.0, 0.25], width=2)
    pl.add_lines(np.asarray(axis_z), connected=False, color=[0.0, 0.5, 1.0], width=2)

    # 视角
    if view_type in view_settings:
        view_angles = [view_settings[view_type]]
    else:
        view_angles = list(view_settings.values())

    parallel_scale = np.linalg.norm(bone_bounds[1] - bone_bounds[0]) * 0.4

    focal_point = np.mean(actor_bounds, axis=0)
    camera_distance = np.linalg.norm(actor_bounds[1] - actor_bounds[0]) * 2.0
    z_axis = np.array([0.0, 0.0, 1.0])

    imgs = []
    for azimuth, elevation in view_angles:
        pl.camera_position = 'xz'
        pl.reset_camera(bounds=np.array([-bone_hl, bone_hl]).T.flatten())
        pl.camera.Azimuth(azimuth)
        pl.camera.Elevation(elevation)
        pl.camera.OrthogonalizeViewUp()
        pl.camera.parallel_scale = parallel_scale
        pl.reset_camera_clipping_range()
        pl.render()

        imgs.append(np.array(pl.screenshot(return_img=True)))

    if len(imgs) == 9:
        img = np.vstack([np.hstack(imgs[i:i + 3]) for i in range(0, 9, 3)])
    else:
        img = imgs[0]

    with views[1]:
        st.image(img, ' '.join(['扫描坐标系', *[f'{_:.2f} mm' for _ in scan_cs[:3]], *[f'{_:.1f} °' for _ in scan_cs[3:]]]))

    pl.close()
    del pl
