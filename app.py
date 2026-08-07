from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import Fullscreen, LocateControl, MousePosition
from streamlit_folium import st_folium


st.set_page_config(
    page_title="토지피복 · S-DoT 지도",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass(frozen=True)
class LandCoverLayer:
    label: str
    layer_name: str
    description: str


@dataclass(frozen=True)
class Metric:
    label: str
    column: str
    unit: str


LAYERS = {
    "1980년대 말": LandCoverLayer("1980년대 말", "EGIS:lv1_1980yr", "대분류 토지피복지도"),
    "1990년대 말": LandCoverLayer("1990년대 말", "EGIS:lv1_1990yr", "대분류 토지피복지도"),
    "2000년대 말": LandCoverLayer("2000년대 말", "EGIS:lv1_2000yr", "대분류 토지피복지도"),
    "2010년대 말": LandCoverLayer("2010년대 말", "EGIS:lv1_2010yr", "대분류 토지피복지도"),
}

METRICS = {
    "온도": Metric("온도", "temperature_c", "℃"),
    "습도": Metric("습도", "humidity_pct", "%"),
    "풍속": Metric("풍속", "wind_speed_ms", "m/s"),
    "조도": Metric("조도", "illuminance_lux", "lux"),
    "자외선": Metric("자외선", "uv_index", "UV"),
    "소음": Metric("소음", "noise_db", "dB"),
    "흑구온도": Metric("흑구온도", "globe_temperature_c", "℃"),
    "이산화질소": Metric("이산화질소", "no2_ppm", "ppm"),
    "일산화탄소": Metric("일산화탄소", "co_ppm", "ppm"),
    "이산화황": Metric("이산화황", "so2_ppm", "ppm"),
    "암모니아": Metric("암모니아", "nh3_ppm", "ppm"),
    "황화수소": Metric("황화수소", "h2s_ppm", "ppm"),
    "오존": Metric("오존", "o3_ppm", "ppm"),
}

PLACES = {
    "서울": ([37.5665, 126.9780], 11),
    "전국": ([36.35, 127.75], 7),
    "수도권": ([37.50, 127.00], 10),
    "부산·울산": ([35.40, 129.15], 10),
    "광주·전남": ([34.95, 126.85], 9),
    "대전·세종": ([36.45, 127.30], 10),
    "제주": ([33.38, 126.55], 10),
}

BASEMAPS = {
    "OpenStreetMap": ("OpenStreetMap", "© OpenStreetMap contributors"),
    "CartoDB 밝은 지도": ("CartoDB positron", "© OpenStreetMap contributors © CARTO"),
    "CartoDB 어두운 지도": ("CartoDB dark_matter", "© OpenStreetMap contributors © CARTO"),
}

WMS_URL = "https://api.mcee.go.kr/geoserver/gwc/service/wms"
LAND_COVER_SOURCE = "https://aid.mcee.go.kr/api/land.do"
SDOT_SOURCE = "https://data.seoul.go.kr/dataList/OA-22833/A/1/datasetView.do"
DATA_FILE_NAME = "sdot_nature_20260720_20260726.parquet"


def find_data_path() -> Path | None:
    app_dir = Path(__file__).resolve().parent
    candidates = (
        app_dir / DATA_FILE_NAME,
        app_dir / "data" / DATA_FILE_NAME,
    )
    return next((path for path in candidates if path.is_file()), None)


@st.cache_data(show_spinner="S-DoT 데이터를 불러오는 중입니다…")
def load_sdot_data(data_path: str) -> pd.DataFrame:
    return pd.read_parquet(data_path)


def build_base_map(
    center: list[float],
    zoom: int,
    basemap_name: str,
    layer: LandCoverLayer,
    opacity: float,
) -> folium.Map:
    tiles, attribution = BASEMAPS[basemap_name]
    map_object = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=tiles,
        attr=attribution,
        control_scale=True,
        prefer_canvas=True,
    )
    folium.WmsTileLayer(
        url=WMS_URL,
        layers=layer.layer_name,
        styles="",
        fmt="image/png",
        transparent=True,
        version="1.1.1",
        name=f"토지피복 · {layer.label}",
        attr="기후에너지환경부 환경공간정보서비스",
        overlay=True,
        control=True,
        show=True,
        opacity=opacity,
        tiled=True,
    ).add_to(map_object)
    Fullscreen(position="topright", title="전체 화면", title_cancel="전체 화면 종료").add_to(map_object)
    LocateControl(position="topright", strings={"title": "내 위치"}).add_to(map_object)
    MousePosition(
        position="bottomright",
        separator=" / ",
        prefix="위도·경도",
        lat_formatter="function(num) {return L.Util.formatNum(num, 5);}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 5);}",
    ).add_to(map_object)
    return map_object


def add_sdot_layer(map_object: folium.Map, data: pd.DataFrame, metric: Metric) -> None:
    measured = data[metric.column].dropna()
    if measured.empty:
        value_min, value_max = 0.0, 1.0
    else:
        value_min = float(measured.quantile(0.05))
        value_max = float(measured.quantile(0.95))
        if value_min == value_max:
            value_max = value_min + 1.0

    color_scale = LinearColormap(
        colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
        vmin=value_min,
        vmax=value_max,
        caption=f"S-DoT {metric.label} ({metric.unit}) · 5–95 백분위 색상범위",
    )
    sensor_group = folium.FeatureGroup(name=f"S-DoT · {metric.label}", show=True)

    for row in data.itertuples(index=False):
        value = getattr(row, metric.column)
        has_value = pd.notna(value)
        color = color_scale(float(value)) if has_value else "#8b949e"
        value_text = f"{float(value):,.2f} {metric.unit}" if has_value else "측정값 없음"
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:230px;line-height:1.55">
          <strong>{escape(str(row.serial))}</strong><br>
          {escape(str(row.address))}<hr style="margin:.45rem 0">
          <b>{escape(metric.label)}:</b> {escape(value_text)}<br>
          <b>측정시각:</b> {row.measured_at:%Y-%m-%d %H:%M}<br>
          <b>지역:</b> {escape(str(row.district))} · {escape(str(row.neighborhood))}
        </div>
        """
        folium.CircleMarker(
            location=[float(row.latitude), float(row.longitude)],
            radius=5.5,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=f"{row.serial} · {value_text}",
            popup=folium.Popup(popup_html, max_width=330),
        ).add_to(sensor_group)

    sensor_group.add_to(map_object)
    if not measured.empty:
        color_scale.add_to(map_object)


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 1rem; max-width: 1500px;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
      .map-kicker {color:#2c7a5a;font-weight:700;letter-spacing:.08em;font-size:.78rem;margin-bottom:.25rem;}
      .map-subtitle {color:#64706d;margin-top:-.65rem;margin-bottom:1rem;}
      .status-card {background:rgba(44,122,90,.08);border:1px solid rgba(44,122,90,.18);
                    border-radius:.75rem;padding:.7rem .85rem;margin:.5rem 0 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="map-kicker">LAND COVER × SEOUL URBAN SENSOR</div>', unsafe_allow_html=True)
st.title("토지피복 · S-DoT 환경지도")
st.markdown(
    '<p class="map-subtitle">환경공간정보서비스 토지피복 위에서 서울 도시데이터 센서 측정값을 탐색합니다.</p>',
    unsafe_allow_html=True,
)

sdot_snapshot: pd.DataFrame | None = None
selected_metric: Metric | None = None
selected_timestamp: pd.Timestamp | None = None

with st.sidebar:
    st.header("지도 설정")
    selected_period = st.selectbox("토지피복 기준 시기", list(LAYERS), index=0)
    selected_place = st.selectbox("빠른 이동", list(PLACES), index=0)
    selected_basemap = st.selectbox("배경지도", list(BASEMAPS), index=1)
    selected_opacity = st.slider("토지피복 투명도", 0.1, 1.0, 0.65, 0.05)

    st.divider()
    st.subheader("S-DoT 센서")
    show_sdot = st.toggle("S-DoT 레이어 표시", value=True)

    if show_sdot:
        data_path = find_data_path()
        if data_path is None:
            st.error("S-DoT 데이터 파일이 저장소에 없습니다.")
            st.code(DATA_FILE_NAME)
            st.info("위 파일을 app.py와 같은 폴더에 업로드한 뒤 앱을 재부팅해 주세요.")
            st.stop()
        sdot_data = load_sdot_data(str(data_path))
        selected_metric_name = st.selectbox("색상 기준 측정항목", list(METRICS), index=0)
        selected_metric = METRICS[selected_metric_name]
        timestamps = sorted(sdot_data["measured_at"].dropna().unique())
        selected_timestamp = st.select_slider(
            "측정시각",
            options=timestamps,
            value=timestamps[-1],
            format_func=lambda value: pd.Timestamp(value).strftime("%m-%d %H:%M"),
        )
        districts = ["전체", *sorted(sdot_data["district"].dropna().astype(str).unique())]
        selected_district = st.selectbox("자치구", districts)

        sdot_snapshot = sdot_data[sdot_data["measured_at"] == selected_timestamp].copy()
        if selected_district != "전체":
            sdot_snapshot = sdot_snapshot[sdot_snapshot["district"].astype(str) == selected_district]
        sdot_snapshot = sdot_snapshot.drop_duplicates("serial", keep="last")

    selected_layer = LAYERS[selected_period]
    st.markdown(
        f"""
        <div class="status-card">
          <strong>{selected_layer.label}</strong><br>
          <small>{selected_layer.description}<br>레이어: {selected_layer.layer_name}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("지도 우측 레이어 메뉴에서도 토지피복과 S-DoT를 각각 켜고 끌 수 있습니다.")

if show_sdot and sdot_snapshot is not None and selected_metric is not None:
    available_values = sdot_snapshot[selected_metric.column].dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("표시 센서", f"{len(sdot_snapshot):,}개")
    col2.metric(
        f"{selected_metric.label} 중앙값",
        f"{available_values.median():,.2f} {selected_metric.unit}" if not available_values.empty else "값 없음",
    )
    col3.metric("측정시각", pd.Timestamp(selected_timestamp).strftime("2026-%m-%d %H:%M"))

center, zoom = PLACES[selected_place]
map_object = build_base_map(
    center=center,
    zoom=zoom,
    basemap_name=selected_basemap,
    layer=selected_layer,
    opacity=selected_opacity,
)

if show_sdot and sdot_snapshot is not None and selected_metric is not None:
    add_sdot_layer(map_object, sdot_snapshot, selected_metric)

folium.LayerControl(collapsed=False, position="topright").add_to(map_object)

map_key_parts = [selected_period, selected_place, selected_basemap, str(selected_opacity), str(show_sdot)]
if selected_timestamp is not None and selected_metric is not None:
    snapshot_size = len(sdot_snapshot) if sdot_snapshot is not None else 0
    map_key_parts.extend([str(selected_timestamp), selected_metric.column, str(snapshot_size)])

st_folium(
    map_object,
    width=None,
    height=680,
    returned_objects=[],
    key="map-" + "-".join(map_key_parts),
)

with st.expander("데이터 및 이용 안내"):
    st.markdown(
        f"""
        - **토지피복 레이어:** `{selected_layer.layer_name}` / EPSG:3857
        - **S-DoT 기간:** 2026-07-20 00:07 ~ 2026-07-26 23:07
        - **S-DoT 규모:** 149,419건, 설치위치가 확인된 센서 954개
        - **토지피복 출처:** [기후에너지환경부 환경공간정보서비스]({LAND_COVER_SOURCE})
        - **S-DoT 출처:** [서울열린데이터광장]({SDOT_SOURCE})

        S-DoT 값은 연구·탐색용 측정자료이며 서울시의 공식 분석 결과가 아닙니다. 통신 지연,
        장비 장애 또는 현장 여건에 따라 결측이나 이상값이 포함될 수 있습니다. 회색 센서는
        선택한 시각·항목의 측정값이 없는 지점입니다.
        """
    )

st.caption("자료 출처: 기후에너지환경부 환경공간정보서비스 · 서울특별시 서울열린데이터광장")
