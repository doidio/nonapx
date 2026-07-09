# uv run streamlit run a3_align_ct_px.py --server.port 8501 -- --config config.toml

import argparse
import pickle
import json
from pathlib import Path
from functools import partial

import streamlit as st
import tomlkit

st.set_page_config('Nonapx', initial_sidebar_state='collapsed', layout='wide')
st.markdown('### Nonapx CT/PX 对齐')

if (it := st.session_state.get('init')) is None:
    with st.spinner('初始化', show_time=True):  # noqa
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='config.toml', type=str)
        parser.add_argument('--max_workers', type=int, default=16)
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

    def fn_patient(_):
        return f'{_} CT {len(pair_meta[_]['CT'])} PX {len(pair_meta[_]['PX'])}'

    patient_id = st.selectbox('Patient ID', list(sorted(pair_meta.keys())), format_func=fn_patient)

    def fn_series(mo, _):
        return ' '.join([
            pair_meta[patient_id][mo][_]['Manufacturer'],
            pair_meta[patient_id][mo][_]['ManufacturerModelName'],
            _,
        ])

    ct_series_id = st.radio('CT Series UID', list(sorted(pair_meta[patient_id]['CT'])), format_func=partial(fn_series, 'CT'))

    st.code(json.dumps(pair_meta[patient_id]['CT'][ct_series_id]), 'toml')

    px_series_id = st.radio('PX Series UID', list(sorted(pair_meta[patient_id]['PX'])), format_func=partial(fn_series, 'PX'))

    st.code(json.dumps(pair_meta[patient_id]['PX'][px_series_id]), 'toml')
