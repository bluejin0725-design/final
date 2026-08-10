from __future__ import annotations

from dataclasses import dataclass
from html import escape
import gzip
import json
from pathlib import Path

import altair as alt
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
    "SHP 영역": ([37.5625, 126.9875], 13),
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
SHP_IMAGE_FILE_NAME = "landcover_detail_2024.png"
SHP_META_FILE_NAME = "landcover_detail_2024.json"
BUFFER_FILE_NAME = "sdot_buffers_300m.geojson.gz"
BUFFER_STATS_FILE_NAME = "sdot_buffer300m_landcover.parquet"


def find_data_path() -> Path | None:
    app_dir = Path(__file__).resolve().parent
    candidates = (
        app_dir / DATA_FILE_NAME,
        app_dir / "data" / DATA_FILE_NAME,
    )
    return next((path for path in candidates if path.is_file()), None)


def find_asset(file_name: str) -> Path | None:
    app_dir = Path(__file__).resolve().parent
    candidates = (app_dir / file_name, app_dir / "data" / file_name)
    return next((path for path in candidates if path.is_file()), None)


@st.cache_data(show_spinner="S-DoT 데이터를 불러오는 중입니다…")
def load_sdot_data(data_path: str) -> pd.DataFrame:
    return pd.read_parquet(data_path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_gzip_json(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


@st.cache_data(show_spinner="300m 버퍼 분석결과를 불러오는 중입니다…")
def load_buffer_stats(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def class_color(class_code: str) -> str:
    code = str(class_code)
    base_hues = {"1": 355, "2": 28, "3": 50, "4": 135, "5": 175, "6": 210, "7": 265}
    base_hue = base_hues.get(code[:1], 195)
    try:
        detail_offset = (int(code) % 100) * 1.7
    except ValueError:
        detail_offset = sum(ord(character) for character in code) % 40
    return f"hsl({(base_hue + detail_offset) % 360:.0f}, 68%, 46%)"


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


def add_shp_overlay(
    map_object: folium.Map,
    image_path: Path,
    metadata: dict,
    opacity: float,
) -> None:
    folium.raster_layers.ImageOverlay(
        image=str(image_path),
        bounds=metadata["bounds"],
        name="SHP · 2024 세분류 토지피복",
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=2,
        pixelated=False,
        show=True,
    ).add_to(map_object)

    legend_rows = "".join(
        f'<div><span style="display:inline-block;width:11px;height:11px;'
        f'background:{item["color"]};margin-right:6px;border-radius:2px"></span>'
        f'{escape(str(item["name"]))}</div>'
        for item in metadata["legend"]
    )
    legend_html = f"""
    <div style="position:fixed;left:12px;bottom:28px;z-index:9999;background:rgba(255,255,255,.92);
                color:#18201e;padding:9px 11px;border-radius:7px;border:1px solid #cdd6d2;
                font:12px/1.5 sans-serif;box-shadow:0 1px 5px rgba(0,0,0,.18)">
      <b>2024 SHP 대분류</b>{legend_rows}
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))


def add_buffer_layer(
    map_object: folium.Map,
    buffer_geojson: dict,
    stats: pd.DataFrame,
    level: str,
    min_coverage: float,
    selected_serial: str | None,
) -> int:
    level_stats = stats[stats["level"] == level].copy()
    dominant_indexes = level_stats.groupby("serial")["share_pct"].idxmax()
    dominant = level_stats.loc[
        dominant_indexes, ["serial", "class_code", "class_name", "share_pct"]
    ].set_index("serial")

    features = []
    for source_feature in buffer_geojson["features"]:
        properties = dict(source_feature["properties"])
        serial = str(properties["serial"])
        coverage = float(properties.get("coverage_pct", 0))
        if coverage < min_coverage or serial not in dominant.index:
            continue
        dominant_row = dominant.loc[serial]
        properties.update(
            {
                "dominant_code": str(dominant_row["class_code"]),
                "dominant_name": str(dominant_row["class_name"]),
                "dominant_share": float(dominant_row["share_pct"]),
                "selected": serial == selected_serial,
            }
        )
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": source_feature["geometry"],
            }
        )

    filtered_geojson = {"type": "FeatureCollection", "features": features}

    def style_function(feature: dict) -> dict:
        properties = feature["properties"]
        selected = properties["selected"]
        color = class_color(properties["dominant_code"])
        return {
            "color": "#111827" if selected else color,
            "fillColor": color,
            "weight": 4 if selected else 2,
            "fillOpacity": 0.35 if selected else 0.18,
            "dashArray": None if properties["coverage_pct"] >= 99 else "5 4",
        }

    layer = folium.GeoJson(
        filtered_geojson,
        name=f"300m 버퍼 · {level}",
        style_function=style_function,
        highlight_function=lambda _: {"weight": 4, "fillOpacity": 0.32},
        show=True,
        control=True,
        tooltip=folium.GeoJsonTooltip(
            fields=["serial", "dominant_name", "dominant_share", "coverage_pct"],
            aliases=["센서", "우세 용도", "우세 비율(%)", "SHP 피복률(%)"],
            localize=True,
            sticky=False,
        ),
        popup=folium.GeoJsonPopup(
            fields=["serial", "address", "dominant_name", "dominant_share", "coverage_pct"],
            aliases=["센서", "주소", "우세 용도", "우세 비율(%)", "SHP 피복률(%)"],
            localize=True,
        ),
    )
    layer.add_to(map_object)
    return len(features)


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

st.markdown('<div class="map-kicker">WMS × SHAPEFILE × SEOUL URBAN SENSOR</div>', unsafe_allow_html=True)
st.title("토지피복 · S-DoT 300m 버퍼 분석")
st.markdown(
    '<p class="map-subtitle">S-DoT 반경 300m 안의 토지피복 용도 구성비를 대·중·세분류별로 탐색합니다.</p>',
    unsafe_allow_html=True,
)

sdot_snapshot: pd.DataFrame | None = None
selected_metric: Metric | None = None
selected_timestamp: pd.Timestamp | None = None
buffer_stats: pd.DataFrame | None = None
buffer_geojson: dict | None = None
selected_buffer_serial: str | None = None
buffer_level = "대분류"
min_buffer_coverage = 50.0

with st.sidebar:
    st.header("지도 설정")
    selected_period = st.selectbox("토지피복 기준 시기", list(LAYERS), index=0)
    selected_place = st.selectbox("빠른 이동", list(PLACES), index=0)
    selected_basemap = st.selectbox("배경지도", list(BASEMAPS), index=1)
    selected_opacity = st.slider("토지피복 투명도", 0.1, 1.0, 0.65, 0.05)

    st.divider()
    st.subheader("세분류 SHP")
    show_shp = st.toggle("2024 SHP 레이어 표시", value=True)
    shp_opacity = st.slider("SHP 투명도", 0.1, 1.0, 0.72, 0.05, disabled=not show_shp)
    shp_image_path = find_asset(SHP_IMAGE_FILE_NAME)
    shp_meta_path = find_asset(SHP_META_FILE_NAME)
    if show_shp and (shp_image_path is None or shp_meta_path is None):
        st.warning("SHP 오버레이 파일 2개가 저장소에 없습니다.")

    st.divider()
    st.subheader("300m 버퍼 분석")
    show_buffers = st.toggle("버퍼 구성비 레이어 표시", value=True)
    buffer_file_path = find_asset(BUFFER_FILE_NAME)
    buffer_stats_path = find_asset(BUFFER_STATS_FILE_NAME)
    if show_buffers and (buffer_file_path is None or buffer_stats_path is None):
        st.warning("300m 버퍼 분석파일 2개가 저장소에 없습니다.")
    elif show_buffers:
        buffer_geojson = load_gzip_json(str(buffer_file_path))
        buffer_stats = load_buffer_stats(str(buffer_stats_path))
        buffer_level = st.selectbox("토지피복 분류단계", ["대분류", "중분류", "세분류"])
        min_buffer_coverage = st.slider(
            "최소 SHP 피복률(%)", 0, 100, 50, 5,
            help="버퍼 전체 면적 중 제공된 SHP가 차지하는 비율입니다.",
        )
        feature_lookup = {
            str(feature["properties"]["serial"]): feature["properties"]
            for feature in buffer_geojson["features"]
            if float(feature["properties"].get("coverage_pct", 0)) >= min_buffer_coverage
        }
        detail_options = ["선택 안 함", *sorted(feature_lookup)]
        selected_buffer_option = st.selectbox(
            "버퍼 상세보기",
            detail_options,
            format_func=lambda serial: (
                serial
                if serial == "선택 안 함"
                else f"{serial} · {feature_lookup[serial].get('address', '')}"
            ),
        )
        if selected_buffer_option != "선택 안 함":
            selected_buffer_serial = selected_buffer_option

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
if selected_buffer_serial and buffer_geojson is not None:
    selected_properties = next(
        feature["properties"]
        for feature in buffer_geojson["features"]
        if str(feature["properties"]["serial"]) == selected_buffer_serial
    )
    center = [float(selected_properties["latitude"]), float(selected_properties["longitude"])]
    zoom = 16
map_object = build_base_map(
    center=center,
    zoom=zoom,
    basemap_name=selected_basemap,
    layer=selected_layer,
    opacity=selected_opacity,
)

if show_shp and shp_image_path is not None and shp_meta_path is not None:
    shp_metadata = load_json(str(shp_meta_path))
    add_shp_overlay(map_object, shp_image_path, shp_metadata, shp_opacity)

visible_buffer_count = 0
if show_buffers and buffer_geojson is not None and buffer_stats is not None:
    visible_buffer_count = add_buffer_layer(
        map_object,
        buffer_geojson,
        buffer_stats,
        buffer_level,
        min_buffer_coverage,
        selected_buffer_serial,
    )

if show_sdot and sdot_snapshot is not None and selected_metric is not None:
    add_sdot_layer(map_object, sdot_snapshot, selected_metric)

folium.LayerControl(collapsed=False, position="topright").add_to(map_object)

map_key_parts = [
    selected_period,
    selected_place,
    selected_basemap,
    str(selected_opacity),
    str(show_shp),
    str(shp_opacity),
    str(show_buffers),
    buffer_level,
    str(min_buffer_coverage),
    str(selected_buffer_serial),
    str(show_sdot),
]
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

if show_buffers and buffer_stats is not None:
    if selected_buffer_serial:
        selected_composition = buffer_stats[
            (buffer_stats["serial"] == selected_buffer_serial)
            & (buffer_stats["level"] == buffer_level)
        ].sort_values("share_pct", ascending=False)
        selected_coverage = float(selected_composition["coverage_pct"].iloc[0])
        dominant_row = selected_composition.iloc[0]
        st.subheader(f"{selected_buffer_serial} · 300m 토지피복 구성비")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("SHP 피복률", f"{selected_coverage:.1f}%")
        metric2.metric("우세 용도", str(dominant_row["class_name"]))
        metric3.metric("우세 비율", f"{float(dominant_row['share_pct']):.1f}%")

        chart_data = selected_composition[["class_code", "class_name", "share_pct", "area_m2"]].copy()
        chart_data["color"] = chart_data["class_code"].map(class_color)
        composition_chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("share_pct:Q", title="구성비 (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("class_name:N", title=None, sort="-x"),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("class_name:N", title="용도"),
                    alt.Tooltip("share_pct:Q", title="구성비(%)", format=".2f"),
                    alt.Tooltip("area_m2:Q", title="면적(㎡)", format=",.0f"),
                ],
            )
            .properties(height=max(180, len(chart_data) * 27))
        )
        st.altair_chart(composition_chart, width="stretch")
        st.caption("구성비는 해당 버퍼에서 SHP로 피복된 면적을 100%로 계산합니다.")
    else:
        st.info(f"현재 조건에서 {visible_buffer_count}개 버퍼가 표시됩니다. 사이드바에서 센서를 선택하면 구성비 차트를 볼 수 있습니다.")

with st.expander("데이터 및 이용 안내"):
    st.markdown(
        f"""
        - **토지피복 레이어:** `{selected_layer.layer_name}` / EPSG:3857
        - **SHP:** 7개 원본 파일, 97,018개 세분류 폴리곤, 약 2m 래스터 해상도
        - **버퍼 분석:** S-DoT 954개 지점의 반경 300m, SHP 중첩 센서 126개
        - **구성비 기준:** 버퍼 내 SHP 피복면적 대비 각 용도 교차면적
        - **S-DoT 기간:** 2026-07-20 00:07 ~ 2026-07-26 23:07
        - **S-DoT 규모:** 149,419건, 설치위치가 확인된 센서 954개
        - **토지피복 출처:** [기후에너지환경부 환경공간정보서비스]({LAND_COVER_SOURCE})
        - **S-DoT 출처:** [서울열린데이터광장]({SDOT_SOURCE})

        S-DoT 값은 연구·탐색용 측정자료이며 서울시의 공식 분석 결과가 아닙니다. 통신 지연,
        장비 장애 또는 현장 여건에 따라 결측이나 이상값이 포함될 수 있습니다. 회색 센서는
        선택한 시각·항목의 측정값이 없는 지점입니다.
        """
    )

st.caption("자료 출처: 기후에너지환경부 환경공간정보서비스 · 사용자 제공 2024 SHP · 서울특별시 서울열린데이터광장")
