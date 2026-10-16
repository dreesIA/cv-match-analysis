"""Interactive showcase for annotated training clips and synchronized actions."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.showcase.data import (
    build_clip_windows,
    default_search_roots,
    discover_bundesliga_clips,
)
from src.showcase.player import build_player_html, video_data_uri


st.set_page_config(
    page_title="updt.pro · Training intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --accent:#73df93; }
      html, body, .stApp, [data-testid="stAppViewContainer"] { background:#000 !important; background-image:none !important; }
      [data-testid="stHeader"] { background:#000; }
      [data-testid="stSidebar"] { background:#03130d; border-right:1px solid #123a2a; }
      .block-container { max-width:1500px; padding-top:1.6rem; padding-bottom:1.25rem; }
      .brand { color:#73df93; font-size:.72rem; font-weight:850; letter-spacing:.18em; text-transform:uppercase; margin:0 0 1rem; }
      div[data-testid="stSelectbox"] label { color:#a8b8b0; }
      .stAlert { background:#051c14; border-color:#18543b; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=4)
def _discover():
    return discover_bundesliga_clips(default_search_roots(Path.cwd()))


@st.cache_data(show_spinner="Preparing tracking video…", max_entries=2)
def _video_uri(path: str, modified_ns: int) -> str:
    del modified_ns  # cache invalidator; path is the actual input
    return video_data_uri(Path(path))


with st.sidebar:
    bundles = _discover()

    if bundles:
        bundle = st.selectbox(
            "Clip",
            bundles,
            format_func=lambda item: item.name,
        )
        clips = build_clip_windows(bundle.events, bundle.duration_s)
        clip = st.selectbox(
            "Moment",
            clips,
            format_func=lambda item: f"{item.label}  ·  {item.detail}",
        )
    else:
        bundle = None
        clip = None

if bundle is None or clip is None:
    st.error("Bundesliga clips not found.")
    st.stop()

stat = bundle.video_path.stat()
uri = _video_uri(str(bundle.video_path), stat.st_mtime_ns)
radar_uri = None
if bundle.radar_path:
    radar_stat = bundle.radar_path.stat()
    radar_uri = _video_uri(str(bundle.radar_path), radar_stat.st_mtime_ns)
player = build_player_html(
    video_uri=uri,
    events=bundle.events,
    clip=clip,
    bundle_name=bundle.name,
    radar_uri=radar_uri,
)
st.iframe(player, height=880, width="stretch", alt="Bundesliga tracking dashboard")
