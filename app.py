from __future__ import annotations

from html import escape
import gzip
import json
from pathlib import Path
from urllib.parse import quote

import altair as alt
import folium
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import Fullscreen, LocateControl
from folium.utilities import JsCode
from streamlit_folium import st_folium


st.set_page_config(
    page_title="S-DoT 300m 토지피복 분석",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLACES = {
    "SHP 영역": ([37.5625, 126.9875], 13),
    "서울": ([37.5665, 126.9780], 11),
}

BASEMAPS = {
    "OpenStreetMap": ("OpenStreetMap", "© OpenStreetMap contributors"),
    "CartoDB 밝은 지도": ("CartoDB positron", "© OpenStreetMap contributors © CARTO"),
    "CartoDB 어두운 지도": ("CartoDB dark_matter", "© OpenStreetMap contributors © CARTO"),
}

LAND_COVER_SOURCE = "https://aid.mcee.go.kr/api/land.do"
SDOT_SOURCE = "https://data.seoul.go.kr/dataList/OA-22833/A/1/datasetView.do"
DATA_FILE_NAME = "sdot_nature_20260720_20260726.parquet"
SHP_IMAGE_FILE_NAME = "landcover_detail_2024.png"
SHP_META_FILE_NAME = "landcover_detail_2024.json"
BUFFER_FILE_NAME = "sdot_buffers_300m.geojson.gz"
BUFFER_STATS_FILE_NAME = "sdot_buffer300m_landcover.parquet"

TEMPERATURE_COLUMN = "temperature_c"
TEMPERATURE_MIN = -30.0
TEMPERATURE_MAX = 50.0
TEMPERATURE_COLORS = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]


def find_asset(file_name: str) -> Path | None:
    app_dir = Path(__file__).resolve().parent
    candidates = (app_dir / file_name, app_dir / "data" / file_name)
    return next((path for path in candidates if path.is_file()), None)


@st.cache_data(show_spinner="S-DoT 데이터를 불러오는 중입니다…")
def load_sdot_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


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


def pie_chart_data_url(
    row: pd.Series,
    composition_columns: list[str],
    colors: dict[str, str],
) -> str:
    """센서별 토지피복 구성비를 표 안에서 표시할 SVG 파이차트로 변환합니다."""
    segments = []
    accumulated = 0.0
    for column in composition_columns:
        raw_share = pd.to_numeric(row.get(column, 0), errors="coerce")
        share = 0.0 if pd.isna(raw_share) else float(raw_share)
        if share <= 0:
            continue
        segments.append(
            '<circle cx="36" cy="36" r="18" fill="none" '
            f'stroke="{colors[column]}" stroke-width="36" pathLength="100" '
            f'stroke-dasharray="{share:.4f} {100 - share:.4f}" '
            f'stroke-dashoffset="{-accumulated:.4f}" transform="rotate(-90 36 36)"/>'
        )
        accumulated += share
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 72 72">'
        '<circle cx="36" cy="36" r="36" fill="#e5e7eb"/>'
        + "".join(segments)
        + '</svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


def clean_temperature(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    temperature = pd.to_numeric(cleaned[TEMPERATURE_COLUMN], errors="coerce")
    cleaned[TEMPERATURE_COLUMN] = temperature.where(
        temperature.between(TEMPERATURE_MIN, TEMPERATURE_MAX)
    )
    return cleaned


def build_base_map(center: list[float], zoom: int, basemap_name: str) -> folium.Map:
    tiles, attribution = BASEMAPS[basemap_name]
    map_object = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=tiles,
        attr=attribution,
        control_scale=True,
        prefer_canvas=True,
    )
    Fullscreen(position="topright", title="전체 화면", title_cancel="전체 화면 종료").add_to(map_object)
    LocateControl(position="topright", strings={"title": "내 위치"}).add_to(map_object)
    return map_object


def add_shp_overlay(
    map_object: folium.Map,
    image_path: Path,
    metadata: dict,
    opacity: float,
) -> None:
    folium.raster_layers.ImageOverlay(
        image=str(image_path),
        bounds=metadata["bounds"],
        name="2024 세분류 토지피복 SHP",
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
    <div style="position:fixed;left:12px;bottom:28px;z-index:9999;background:rgba(255,255,255,.93);
                color:#18201e;padding:9px 11px;border-radius:7px;border:1px solid #cdd6d2;
                font:12px/1.5 sans-serif;box-shadow:0 1px 5px rgba(0,0,0,.18)">
      <b>2024 SHP 대분류</b>{legend_rows}
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))


def add_buffer_layer(
    parent_group: folium.FeatureGroup,
    buffer_geojson: dict,
    stats: pd.DataFrame,
    level: str,
    min_coverage: float,
    allowed_serials: set[str],
    selected_serial: str | None,
) -> int:
    level_stats = stats[stats["level"] == level].copy()
    dominant_indexes = level_stats.groupby("serial")["share_pct"].idxmax()
    dominant = level_stats.loc[
        dominant_indexes, ["serial", "class_code", "class_name", "share_pct"]
    ].set_index("serial")
    composition_by_serial = {
        str(serial): [
            {
                "name": escape(str(row.class_name)),
                "share": round(float(row.share_pct), 2),
                "color": class_color(str(row.class_code)),
            }
            for row in group.sort_values("share_pct", ascending=False).itertuples(index=False)
            if float(row.share_pct) > 0
        ]
        for serial, group in level_stats.groupby("serial", sort=False)
    }

    features = []
    for source_feature in buffer_geojson["features"]:
        properties = dict(source_feature["properties"])
        serial = str(properties["serial"])
        coverage = float(properties.get("coverage_pct", 0))
        if (
            coverage < min_coverage
            or serial not in dominant.index
            or serial not in allowed_serials
        ):
            continue
        dominant_row = dominant.loc[serial]
        properties.update(
            {
                "dominant_code": str(dominant_row["class_code"]),
                "dominant_name": str(dominant_row["class_name"]),
                "dominant_share": round(float(dominant_row["share_pct"]), 2),
                "coverage_pct": round(coverage, 2),
                "composition_level": level,
                "composition": composition_by_serial.get(serial, []),
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

    def style_function(feature: dict) -> dict:
        properties = feature["properties"]
        selected = properties["selected"]
        color = class_color(properties["dominant_code"])
        return {
            "color": "#111827" if selected else color,
            "fillColor": color,
            "weight": 4 if selected else 2,
            "fillOpacity": 0.34 if selected else 0.17,
            "dashArray": None if properties["coverage_pct"] >= 99 else "5 4",
        }

    pie_tooltip = JsCode(
        """
        function(feature, layer) {
            const properties = feature.properties || {};
            const items = properties.composition || [];
            let accumulated = 0;
            const colorStops = items.map(function(item) {
                const start = accumulated;
                accumulated = Math.min(100, accumulated + Number(item.share || 0));
                return item.color + " " + start.toFixed(2) + "% " + accumulated.toFixed(2) + "%";
            });
            const pieBackground = colorStops.length
                ? "conic-gradient(" + colorStops.join(",") + ")"
                : "#d1d5db";
            const legendRows = items.map(function(item) {
                return '<div style="display:grid;grid-template-columns:12px 1fr auto;gap:7px;'
                    + 'align-items:center;margin:3px 0">'
                    + '<span style="width:11px;height:11px;border-radius:2px;background:'
                    + item.color + '"></span>'
                    + '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                    + item.name + '</span>'
                    + '<b style="font-variant-numeric:tabular-nums">'
                    + Number(item.share).toFixed(1) + '%</b></div>';
            }).join("");
            const tooltipHtml = '<div style="width:310px;color:#18201e;font:12px/1.4 sans-serif">'
                + '<div style="font-size:13px;font-weight:700;margin-bottom:7px">'
                + properties.serial + ' · 300m ' + properties.composition_level + ' 구성비</div>'
                + '<div style="display:grid;grid-template-columns:112px 1fr;gap:12px;align-items:center">'
                + '<div style="width:108px;height:108px;border-radius:50%;background:'
                + pieBackground + ';box-shadow:inset 0 0 0 1px rgba(0,0,0,.12)"></div>'
                + '<div style="max-height:180px;overflow-y:auto;padding-right:3px">'
                + legendRows + '</div></div>'
                + '<div style="margin-top:7px;padding-top:6px;border-top:1px solid #d7dfdc;'
                + 'color:#59635f">SHP 피복률 '
                + Number(properties.coverage_pct).toFixed(1) + '%</div></div>';
            layer.bindTooltip(tooltipHtml, {
                sticky: true,
                direction: "auto",
                opacity: 0.97,
                maxWidth: 350,
                className: "landcover-pie-tooltip"
            });
        }
        """
    )

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="300m 버퍼",
        style_function=style_function,
        highlight_function=lambda _: {"weight": 4, "fillOpacity": 0.30},
        show=True,
        control=False,
        on_each_feature=pie_tooltip,
        popup=folium.GeoJsonPopup(
            fields=["serial", "address", "coverage_pct"],
            aliases=["센서", "주소", "SHP 피복률(%)"],
            localize=True,
        ),
    ).add_to(parent_group)
    return len(features)


def add_temperature_points(
    parent_group: folium.FeatureGroup,
    map_object: folium.Map,
    data: pd.DataFrame,
) -> None:
    data = data.dropna(subset=[TEMPERATURE_COLUMN]).copy()
    measured = data[TEMPERATURE_COLUMN].dropna()
    if measured.empty:
        value_min, value_max = 0.0, 1.0
    else:
        value_min = float(measured.quantile(0.05))
        value_max = float(measured.quantile(0.95))
        if value_min == value_max:
            value_max = value_min + 1.0
    color_scale = LinearColormap(TEMPERATURE_COLORS, vmin=value_min, vmax=value_max)

    for row in data.itertuples(index=False):
        value = getattr(row, TEMPERATURE_COLUMN)
        has_value = pd.notna(value)
        color = color_scale(float(value)) if has_value else "#8b949e"
        value_text = f"{float(value):,.2f} ℃" if has_value else "유효 측정값 없음"
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:230px;line-height:1.55">
          <strong>{escape(str(row.serial))}</strong><br>
          {escape(str(row.address))}<hr style="margin:.45rem 0">
          <b>온도:</b> {escape(value_text)}<br>
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
            fill_opacity=0.92,
            tooltip=f"{row.serial} · {value_text}",
            popup=folium.Popup(popup_html, max_width=330),
        ).add_to(parent_group)

    if not measured.empty:
        gradient = ",".join(TEMPERATURE_COLORS)
        legend_html = f"""
        <div style="position:fixed;right:14px;bottom:42px;z-index:9999;width:210px;
                    background:rgba(255,255,255,.94);color:#18201e;padding:9px 11px;
                    border-radius:7px;border:1px solid #cdd6d2;font:12px/1.35 sans-serif;
                    box-shadow:0 1px 5px rgba(0,0,0,.18)">
          <b>S-DoT 온도 (℃)</b>
          <div style="height:11px;margin:6px 0 2px;background:linear-gradient(to right,{gradient});"></div>
          <div style="display:flex;justify-content:space-between">
            <span>{value_min:.1f}</span><span>5–95 백분위</span><span>{value_max:.1f}</span>
          </div>
        </div>
        """
        map_object.get_root().html.add_child(folium.Element(legend_html))


def build_sensor_table(
    stats: pd.DataFrame,
    snapshot: pd.DataFrame,
    min_coverage: float,
) -> tuple[pd.DataFrame, list[str]]:
    snapshot = snapshot.dropna(subset=[TEMPERATURE_COLUMN]).copy()
    large_class = stats[stats["level"] == "대분류"].copy()
    large_class = large_class[large_class["coverage_pct"] >= min_coverage]
    composition = large_class.pivot_table(
        index="serial",
        columns="class_name",
        values="share_pct",
        aggfunc="sum",
        fill_value=0,
    )
    composition.columns = [f"{column}(%)" for column in composition.columns]
    composition_columns = list(composition.columns)
    class_colors = {
        f"{row.class_name}(%)": class_color(str(row.class_code))
        for row in large_class.drop_duplicates("class_name").itertuples(index=False)
    }

    dominant_indexes = large_class.groupby("serial")["share_pct"].idxmax()
    dominant = large_class.loc[
        dominant_indexes, ["serial", "class_name", "share_pct", "coverage_pct"]
    ].rename(
        columns={
            "class_name": "우세 용도",
            "share_pct": "우세 비율(%)",
            "coverage_pct": "SHP 피복률(%)",
        }
    )
    sensor_columns = [
        "serial",
        "district",
        "neighborhood",
        "address",
        TEMPERATURE_COLUMN,
    ]
    table = (
        snapshot[sensor_columns]
        .merge(dominant, on="serial", how="inner")
        .merge(composition, on="serial", how="left")
        .rename(
            columns={
                "serial": "센서",
                "district": "자치구",
                "neighborhood": "행정동",
                "address": "주소",
                TEMPERATURE_COLUMN: "온도(℃)",
            }
        )
    )
    numeric_columns = ["온도(℃)", "SHP 피복률(%)", "우세 비율(%)", *composition_columns]
    table[numeric_columns] = table[numeric_columns].round(2)
    pie_column_position = table.columns.get_loc("온도(℃)") + 1
    table.insert(
        pie_column_position,
        "피복 구성비",
        table.apply(
            pie_chart_data_url,
            axis=1,
            composition_columns=composition_columns,
            colors=class_colors,
        ),
    )
    return table, composition_columns


def build_trend_table(sensor_table: pd.DataFrame, composition_columns: list[str]) -> pd.DataFrame:
    valid = sensor_table.dropna(subset=["온도(℃)"]).copy()
    rows = []
    for column in composition_columns:
        shares = valid[column].astype(float)
        temperatures = valid["온도(℃)"].astype(float)
        sample_size = len(valid)
        group_size = max(1, sample_size // 4)
        high_indexes = shares.nlargest(group_size).index
        low_indexes = shares.nsmallest(group_size).index
        correlation = shares.corr(temperatures) if shares.nunique() > 1 else float("nan")
        high_temperature = temperatures.loc[high_indexes].mean()
        low_temperature = temperatures.loc[low_indexes].mean()
        temperature_gap = high_temperature - low_temperature
        if pd.isna(correlation) or abs(correlation) < 0.2:
            tendency = "뚜렷하지 않음"
        elif correlation > 0:
            tendency = "구성비 증가 시 온도 상승 경향"
        else:
            tendency = "구성비 증가 시 온도 하락 경향"
        rows.append(
            {
                "토지피복 용도": column.removesuffix("(%)"),
                "분석 센서수": sample_size,
                "해당 용도 포함 센서수": int((shares > 0).sum()),
                "평균 구성비(%)": shares.mean(),
                "온도 상관계수": correlation,
                "구성비 상위25% 평균온도(℃)": high_temperature,
                "구성비 하위25% 평균온도(℃)": low_temperature,
                "온도차(℃)": temperature_gap,
                "경향": tendency,
            }
        )
    trend = pd.DataFrame(rows)
    numeric = [
        "평균 구성비(%)",
        "온도 상관계수",
        "구성비 상위25% 평균온도(℃)",
        "구성비 하위25% 평균온도(℃)",
        "온도차(℃)",
    ]
    trend[numeric] = trend[numeric].round(3)
    return trend


def build_dominant_use_regression(
    sensor_table: pd.DataFrame,
    min_group_size: int = 3,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    """우세용도 범주별 온도 차이를 기준집단 대비 OLS 계수로 추정합니다."""
    valid = sensor_table[["우세 용도", "온도(℃)"]].dropna().copy()
    valid["우세 용도"] = valid["우세 용도"].astype(str)
    valid["온도(℃)"] = pd.to_numeric(valid["온도(℃)"], errors="coerce")
    valid = valid.dropna(subset=["온도(℃)"])

    group_counts = valid["우세 용도"].value_counts()
    eligible_counts = group_counts[group_counts >= min_group_size]
    excluded = group_counts[group_counts < min_group_size].to_dict()
    if len(eligible_counts) < 2:
        return None, {
            "reason": f"센서가 {min_group_size}개 이상인 우세용도가 2개 이상 필요합니다.",
            "excluded": excluded,
        }

    reference = str(eligible_counts.index[0])
    categories = [reference, *sorted(str(item) for item in eligible_counts.index[1:])]
    analysis = valid[valid["우세 용도"].isin(categories)].copy()

    design = pd.DataFrame({"const": 1.0}, index=analysis.index)
    parameter_names: dict[str, str] = {}
    for index, category in enumerate(categories[1:], start=1):
        parameter = f"dominant_use_{index}"
        parameter_names[category] = parameter
        design[parameter] = (analysis["우세 용도"] == category).astype(float)

    model = sm.OLS(analysis["온도(℃)"].astype(float), design).fit(
        cov_type="HC3",
        use_t=True,
    )
    confidence = model.conf_int(alpha=0.05)
    rows = []
    for category in categories:
        group = analysis[analysis["우세 용도"] == category]["온도(℃)"]
        if category == reference:
            coefficient = 0.0
            standard_error = float("nan")
            lower = float("nan")
            upper = float("nan")
            p_value = float("nan")
            result = "기준집단"
        else:
            parameter = parameter_names[category]
            coefficient = float(model.params[parameter])
            standard_error = float(model.bse[parameter])
            lower = float(confidence.loc[parameter, 0])
            upper = float(confidence.loc[parameter, 1])
            p_value = float(model.pvalues[parameter])
            if p_value < 0.05:
                result = "기준보다 유의하게 높음" if coefficient > 0 else "기준보다 유의하게 낮음"
            else:
                result = "유의한 차이 없음"
        rows.append(
            {
                "우세용도": category,
                "센서수": int(len(group)),
                "평균온도(℃)": float(group.mean()),
                "기준 대비 온도차(℃)": coefficient,
                "표준오차": standard_error,
                "95% 하한(℃)": lower,
                "95% 상한(℃)": upper,
                "p값": p_value,
                "판정": result,
            }
        )

    result_table = pd.DataFrame(rows)
    numeric_columns = [
        "평균온도(℃)",
        "기준 대비 온도차(℃)",
        "표준오차",
        "95% 하한(℃)",
        "95% 상한(℃)",
        "p값",
    ]
    result_table[numeric_columns] = result_table[numeric_columns].round(4)
    overall_p = float(model.f_pvalue) if pd.notna(model.f_pvalue) else float("nan")
    return result_table, {
        "reference": reference,
        "nobs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "overall_p": overall_p,
        "excluded": excluded,
        "min_group_size": min_group_size,
    }


st.markdown(
    """
    <style>
      .block-container {padding-top:1.4rem;padding-bottom:1rem;max-width:1500px;}
      [data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.18);}
      .map-kicker {color:#2c7a5a;font-weight:700;letter-spacing:.08em;font-size:.78rem;margin-bottom:.25rem;}
      .map-subtitle {color:#64706d;margin-top:-.65rem;margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="map-kicker">LAND COVER × 300M BUFFER × TEMPERATURE</div>', unsafe_allow_html=True)
st.title("S-DoT 300m 토지피복 · 온도 분석")
st.markdown(
    '<p class="map-subtitle">온도센서 주변 토지피복 구성비를 비교하고 구성비에 따른 온도 경향을 살펴봅니다.</p>',
    unsafe_allow_html=True,
)

sdot_snapshot: pd.DataFrame | None = None
buffer_stats: pd.DataFrame | None = None
buffer_geojson: dict | None = None
sensor_table: pd.DataFrame | None = None
selected_buffer_serial: str | None = None
selected_timestamp: pd.Timestamp | None = None
buffer_level = "대분류"
min_buffer_coverage = 50.0
selected_district = "전체"

with st.sidebar:
    st.header("지도 설정")
    selected_place = st.selectbox("빠른 이동", list(PLACES), index=0)
    selected_basemap = st.selectbox("배경지도", list(BASEMAPS), index=1)

    st.divider()
    st.subheader("토지피복 SHP")
    show_shp = st.toggle("2024 세분류 토지피복", value=True)
    shp_opacity = st.slider("토지피복 투명도", 0.1, 1.0, 0.72, 0.05, disabled=not show_shp)
    shp_image_path = find_asset(SHP_IMAGE_FILE_NAME)
    shp_meta_path = find_asset(SHP_META_FILE_NAME)
    if show_shp and (shp_image_path is None or shp_meta_path is None):
        st.warning("토지피복 오버레이 파일 2개가 저장소에 없습니다.")

    st.divider()
    st.subheader("온도센서 · 300m 버퍼")
    show_temperature_sensors = st.toggle("S-DoT 온도센서", value=True)

    sdot_path = find_asset(DATA_FILE_NAME)
    buffer_path = find_asset(BUFFER_FILE_NAME)
    buffer_stats_path = find_asset(BUFFER_STATS_FILE_NAME)
    missing_analysis_files = [
        name
        for name, path in [
            (DATA_FILE_NAME, sdot_path),
            (BUFFER_FILE_NAME, buffer_path),
            (BUFFER_STATS_FILE_NAME, buffer_stats_path),
        ]
        if path is None
    ]
    if show_temperature_sensors and missing_analysis_files:
        st.error("분석파일이 저장소에 없습니다: " + ", ".join(missing_analysis_files))
        st.stop()

    if show_temperature_sensors:
        sdot_data = load_sdot_data(str(sdot_path))
        buffer_geojson = load_gzip_json(str(buffer_path))
        buffer_stats = load_buffer_stats(str(buffer_stats_path))

        timestamps = sorted(sdot_data["measured_at"].dropna().unique())
        selected_timestamp = st.select_slider(
            "측정시각",
            options=timestamps,
            value=timestamps[-1],
            format_func=lambda value: pd.Timestamp(value).strftime("%m-%d %H:%M"),
        )
        districts = ["전체", *sorted(sdot_data["district"].dropna().astype(str).unique())]
        selected_district = st.selectbox("자치구", districts)
        buffer_level = st.selectbox("버퍼 토지피복 분류", ["대분류", "중분류", "세분류"])
        min_buffer_coverage = st.slider(
            "최소 SHP 피복률(%)",
            0,
            100,
            50,
            5,
            help="버퍼 전체 면적 중 제공된 SHP가 차지하는 비율입니다.",
        )

        sdot_snapshot = sdot_data[sdot_data["measured_at"] == selected_timestamp].copy()
        if selected_district != "전체":
            sdot_snapshot = sdot_snapshot[
                sdot_snapshot["district"].astype(str) == selected_district
            ]
        sdot_snapshot = clean_temperature(
            sdot_snapshot.drop_duplicates("serial", keep="last")
        )
        # 온도가 비어 있거나 유효범위를 벗어난 센서는 지도·버퍼·표·분석에서 제외합니다.
        sdot_snapshot = sdot_snapshot.dropna(subset=[TEMPERATURE_COLUMN]).copy()
        allowed_serials = set(sdot_snapshot["serial"].astype(str))

        feature_lookup = {
            str(feature["properties"]["serial"]): feature["properties"]
            for feature in buffer_geojson["features"]
            if float(feature["properties"].get("coverage_pct", 0)) >= min_buffer_coverage
            and str(feature["properties"]["serial"]) in allowed_serials
        }
        detail_options = ["선택 안 함", *sorted(feature_lookup)]
        selected_option = st.selectbox(
            "센서 상세보기",
            detail_options,
            format_func=lambda serial: (
                serial
                if serial == "선택 안 함"
                else f"{serial} · {feature_lookup[serial].get('address', '')}"
            ),
        )
        if selected_option != "선택 안 함":
            selected_buffer_serial = selected_option

    st.caption("지도 레이어 메뉴에서 토지피복과 온도센서를 각각 켜고 끌 수 있습니다.")

if show_temperature_sensors and sdot_snapshot is not None:
    valid_temperatures = sdot_snapshot[TEMPERATURE_COLUMN].dropna()
    status1, status2, status3 = st.columns(3)
    status1.metric("표시 센서", f"{len(sdot_snapshot):,}개")
    status2.metric(
        "유효 온도 중앙값",
        f"{valid_temperatures.median():.2f} ℃" if not valid_temperatures.empty else "값 없음",
    )
    status3.metric("측정시각", pd.Timestamp(selected_timestamp).strftime("2026-%m-%d %H:%M"))

center, zoom = PLACES[selected_place]
if selected_buffer_serial and buffer_geojson is not None:
    selected_properties = next(
        feature["properties"]
        for feature in buffer_geojson["features"]
        if str(feature["properties"]["serial"]) == selected_buffer_serial
    )
    center = [float(selected_properties["latitude"]), float(selected_properties["longitude"])]
    zoom = 16

map_object = build_base_map(center, zoom, selected_basemap)

if show_shp and shp_image_path is not None and shp_meta_path is not None:
    add_shp_overlay(
        map_object,
        shp_image_path,
        load_json(str(shp_meta_path)),
        shp_opacity,
    )

visible_buffer_count = 0
if (
    show_temperature_sensors
    and sdot_snapshot is not None
    and buffer_geojson is not None
    and buffer_stats is not None
):
    sensor_group = folium.FeatureGroup(name="S-DoT 온도센서", show=True, control=True)
    allowed_serials = set(sdot_snapshot["serial"].astype(str))
    visible_buffer_count = add_buffer_layer(
        sensor_group,
        buffer_geojson,
        buffer_stats,
        buffer_level,
        min_buffer_coverage,
        allowed_serials,
        selected_buffer_serial,
    )
    add_temperature_points(sensor_group, map_object, sdot_snapshot)
    sensor_group.add_to(map_object)

folium.LayerControl(collapsed=False, position="topright").add_to(map_object)

map_key_parts = [
    selected_place,
    selected_basemap,
    str(show_shp),
    str(shp_opacity),
    str(show_temperature_sensors),
    buffer_level,
    str(min_buffer_coverage),
    str(selected_buffer_serial),
    str(selected_timestamp),
    selected_district,
]
st_folium(
    map_object,
    width=None,
    height=680,
    returned_objects=[],
    key="map-" + "-".join(map_key_parts),
)

if show_temperature_sensors and buffer_stats is not None and sdot_snapshot is not None:
    sensor_table, composition_columns = build_sensor_table(
        buffer_stats,
        sdot_snapshot,
        min_buffer_coverage,
    )

    st.subheader("온도센서별 300m 대분류 구성비")
    sort_col1, sort_col2 = st.columns([2, 1])
    sort_options = [
        "온도(℃)",
        "SHP 피복률(%)",
        "우세 비율(%)",
        *composition_columns,
    ]
    with sort_col1:
        sensor_sort_column = st.selectbox("센서표 정렬 기준", sort_options)
    with sort_col2:
        sensor_sort_ascending = st.toggle("오름차순", value=False, key="sensor_sort_order")
    sorted_sensor_table = sensor_table.sort_values(
        sensor_sort_column,
        ascending=sensor_sort_ascending,
        na_position="last",
    )
    st.dataframe(
        sorted_sensor_table,
        hide_index=True,
        width="stretch",
        height=430,
        row_height=64,
        column_config={
            "피복 구성비": st.column_config.ImageColumn(
                "피복 구성비",
                width="small",
                help="각 센서 300m 버퍼의 대분류 구성비입니다. 두 번 클릭하면 확대됩니다.",
                pinned=True,
            ),
            "온도(℃)": st.column_config.NumberColumn(format="%.2f ℃"),
            "SHP 피복률(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "우세 비율(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            **{
                column: st.column_config.NumberColumn(format="%.1f%%")
                for column in composition_columns
            },
        },
    )
    st.download_button(
        "정렬된 센서표 CSV 다운로드",
        sorted_sensor_table.drop(columns=["피복 구성비"])
        .to_csv(index=False)
        .encode("utf-8-sig"),
        file_name="sdot_300m_landcover_temperature.csv",
        mime="text/csv",
    )

    if selected_buffer_serial:
        selected_composition = buffer_stats[
            (buffer_stats["serial"] == selected_buffer_serial)
            & (buffer_stats["level"] == buffer_level)
        ].sort_values("share_pct", ascending=False)
        if not selected_composition.empty:
            selected_coverage = float(selected_composition["coverage_pct"].iloc[0])
            dominant_row = selected_composition.iloc[0]
            st.subheader(f"{selected_buffer_serial} · 300m 상세 구성비")
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("SHP 피복률", f"{selected_coverage:.1f}%")
            metric2.metric("우세 용도", str(dominant_row["class_name"]))
            metric3.metric("우세 비율", f"{float(dominant_row['share_pct']):.1f}%")
            chart_data = selected_composition[
                ["class_code", "class_name", "share_pct", "area_m2"]
            ].copy()
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

    st.subheader("토지피복 구성비에 따른 온도 경향")
    trend_table = build_trend_table(sensor_table, composition_columns)
    trend_col1, trend_col2 = st.columns([2, 1])
    trend_sort_options = [
        "온도차(℃)",
        "온도 상관계수",
        "평균 구성비(%)",
        "해당 용도 포함 센서수",
    ]
    with trend_col1:
        trend_sort_column = st.selectbox("경향표 정렬 기준", trend_sort_options)
    with trend_col2:
        trend_sort_ascending = st.toggle("오름차순", value=False, key="trend_sort_order")
    sorted_trend = trend_table.sort_values(
        trend_sort_column,
        ascending=trend_sort_ascending,
        na_position="last",
    )
    st.dataframe(
        sorted_trend,
        hide_index=True,
        width="stretch",
        column_config={
            "평균 구성비(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "온도 상관계수": st.column_config.NumberColumn(format="%.3f"),
            "구성비 상위25% 평균온도(℃)": st.column_config.NumberColumn(format="%.2f ℃"),
            "구성비 하위25% 평균온도(℃)": st.column_config.NumberColumn(format="%.2f ℃"),
            "온도차(℃)": st.column_config.NumberColumn(format="%+.2f ℃"),
        },
    )
    st.caption(
        f"현재 시각의 유효 온도 센서 {sensor_table['온도(℃)'].notna().sum()}개를 사용했습니다. "
        "상관과 온도차는 탐색적 연관성이며 인과관계를 뜻하지 않습니다."
    )
else:
    st.info("S-DoT 온도센서 토글을 켜면 센서별 구성비와 온도 경향표가 표시됩니다.")

with st.expander("데이터 및 분석 기준"):
    st.markdown(
        f"""
        - **SHP:** 7개 원본 파일, 97,018개 세분류 폴리곤, 약 2m 래스터 해상도
        - **버퍼:** S-DoT 위치 중심 반경 300m
        - **구성비:** 버퍼 내 SHP 피복면적 대비 각 용도 교차면적
        - **온도 유효범위:** {TEMPERATURE_MIN:.0f}℃ 이상 {TEMPERATURE_MAX:.0f}℃ 이하
        - **S-DoT 기간:** 2026-07-20 00:07 ~ 2026-07-26 23:07
        - **토지피복 출처:** [기후에너지환경부 환경공간정보서비스]({LAND_COVER_SOURCE})
        - **S-DoT 출처:** [서울열린데이터광장]({SDOT_SOURCE})

        SHP가 제공되지 않은 지역은 피복률로 구분합니다. S-DoT 값은 연구·탐색용 자료이며
        통신 지연, 장비 장애 또는 현장 여건에 따른 결측·이상값이 포함될 수 있습니다.
        """
    )

st.caption("자료 출처: 사용자 제공 2024 토지피복 SHP · 서울특별시 서울열린데이터광장")

st.divider()
st.subheader("우세용도에 따른 온도 차이 OLS 회귀분석")
if show_temperature_sensors and sensor_table is not None and not sensor_table.empty:
    regression_table, regression_info = build_dominant_use_regression(sensor_table)
    if regression_table is None:
        st.warning(str(regression_info["reason"]))
    else:
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("분석 센서", f"{int(regression_info['nobs']):,}개")
        metric2.metric("기준 우세용도", str(regression_info["reference"]))
        metric3.metric("설명력 R²", f"{float(regression_info['r_squared']):.3f}")
        overall_p = float(regression_info["overall_p"])
        metric4.metric("모형 전체 p값", f"{overall_p:.4f}" if pd.notna(overall_p) else "계산 불가")

        coefficient_data = regression_table[
            (regression_table["판정"] != "기준집단")
            & regression_table["기준 대비 온도차(℃)"].notna()
        ].copy()
        coefficient_data = coefficient_data.sort_values("기준 대비 온도차(℃)")
        if not coefficient_data.empty:
            category_order = coefficient_data["우세용도"].tolist()
            base = alt.Chart(coefficient_data).encode(
                y=alt.Y("우세용도:N", title=None, sort=category_order),
            )
            confidence_chart = base.mark_rule(strokeWidth=3).encode(
                x=alt.X("95% 하한(℃):Q", title="기준 우세용도 대비 온도차 (℃)"),
                x2="95% 상한(℃):Q",
            )
            point_chart = base.mark_point(filled=True, size=110).encode(
                x=alt.X("기준 대비 온도차(℃):Q", title="기준 우세용도 대비 온도차 (℃)"),
                color=alt.condition(
                    alt.datum["기준 대비 온도차(℃)"] >= 0,
                    alt.value("#d95f02"),
                    alt.value("#1b9e77"),
                ),
                tooltip=[
                    alt.Tooltip("우세용도:N"),
                    alt.Tooltip("센서수:Q"),
                    alt.Tooltip("평균온도(℃):Q", format=".2f"),
                    alt.Tooltip("기준 대비 온도차(℃):Q", format="+.3f"),
                    alt.Tooltip("95% 하한(℃):Q", format="+.3f"),
                    alt.Tooltip("95% 상한(℃):Q", format="+.3f"),
                    alt.Tooltip("p값:Q", format=".4f"),
                ],
            )
            zero_line = (
                alt.Chart(pd.DataFrame({"기준선": [0.0]}))
                .mark_rule(color="#7d8682", strokeDash=[4, 4])
                .encode(x="기준선:Q")
            )
            st.altair_chart(
                (zero_line + confidence_chart + point_chart).properties(
                    height=max(180, len(coefficient_data) * 46)
                ),
                width="stretch",
            )

        st.dataframe(
            regression_table,
            hide_index=True,
            width="stretch",
            column_config={
                "평균온도(℃)": st.column_config.NumberColumn(format="%.2f ℃"),
                "기준 대비 온도차(℃)": st.column_config.NumberColumn(format="%+.3f ℃"),
                "표준오차": st.column_config.NumberColumn(format="%.3f"),
                "95% 하한(℃)": st.column_config.NumberColumn(format="%+.3f ℃"),
                "95% 상한(℃)": st.column_config.NumberColumn(format="%+.3f ℃"),
                "p값": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        excluded = regression_info["excluded"]
        excluded_text = (
            ", ".join(f"{name}({count}개)" for name, count in excluded.items())
            if excluded
            else "없음"
        )
        st.caption(
            f"종속변수는 현재 선택 시각의 센서 온도이며, 기준집단은 "
            f"'{regression_info['reference']}'입니다. 계수와 95% 신뢰구간은 HC3 강건 표준오차를 사용했습니다. "
            f"표본이 {regression_info['min_group_size']}개 미만이라 제외된 우세용도: {excluded_text}. "
            "p<0.05는 기준집단과의 통계적 차이를 뜻하지만 인과효과를 의미하지 않습니다."
        )
else:
    st.info("S-DoT 온도센서 토글을 켜면 우세용도별 OLS 회귀분석이 표시됩니다.")
