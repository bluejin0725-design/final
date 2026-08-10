# 토지피복 · S-DoT 300m 버퍼 분석

기후에너지환경부 환경공간정보서비스 WMS 토지피복지도 위에 서울특별시 S-DoT
환경센서 측정값을 표시하는 Streamlit 앱입니다.

## 주요 기능

- 1980·1990·2000·2010년대 말 대분류 토지피복지도 전환
- S-DoT 레이어 ON/OFF 토글
- 사용자 제공 2024년 세분류 토지피복 SHP 레이어 ON/OFF와 투명도 조절
- SHP 7개, 97,018개 폴리곤을 약 2m 해상도 투명 오버레이로 경량화
- S-DoT 954개 지점에 반경 300m 버퍼 생성
- 버퍼별 대·중·세분류 토지피복 면적과 구성비 사전 계산
- 우세 용도 색상 버퍼, 최소 SHP 피복률 필터, 센서 검색
- 선택한 버퍼의 용도별 구성비 막대그래프와 면적 툴팁
- 2026-07-20~07-26의 시간대별 측정값 탐색
- 온도·습도·풍속·조도·자외선·소음·흑구온도·가스류 측정항목 선택
- 자치구 필터와 측정값 기반 포인트 색상 표시
- 센서 클릭 시 시리얼, 주소, 측정시각 및 측정값 확인

## 프로젝트 구조

```text
.
├── app.py
├── sdot_nature_20260720_20260726.parquet
├── landcover_detail_2024.png
├── landcover_detail_2024.json
├── sdot_buffers_300m.geojson.gz
├── sdot_buffer300m_landcover.parquet
├── requirements.txt
├── README.md
└── .gitignore
```

원본 S-DoT CSV는 53MB이지만, 앱에 필요한 측정값과 공식 설치좌표만 보존한 압축
Parquet 파일은 약 1MB입니다. 149,419개 측정행과 954개 센서의 전체 기간을 유지합니다.

> **중요:** GitHub에 업로드할 때 `app.py`와
> `sdot_nature_20260720_20260726.parquet`를 반드시 같은 폴더에 넣어야 합니다.

SHP 레이어를 표시하려면 `landcover_detail_2024.png`와 `landcover_detail_2024.json`도
`app.py`와 같은 폴더에 업로드해야 합니다.

300m 분석을 사용하려면 `sdot_buffers_300m.geojson.gz`와
`sdot_buffer300m_landcover.parquet`도 같은 폴더에 업로드해야 합니다.

## 버퍼 구성비 해석

- 버퍼 면적은 S-DoT 위치를 기준으로 한 반경 300m 원입니다.
- SHP 피복률은 전체 버퍼 면적 중 제공된 SHP와 겹치는 면적의 비율입니다.
- 용도 구성비는 SHP로 피복된 면적을 100%로 두고 계산합니다.
- 제공된 SHP 영역과 겹치는 센서는 126개이며, 101개는 피복률이 99% 이상입니다.

## 로컬 실행

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## GitHub 및 Streamlit Community Cloud 배포

GitHub 저장소에 프로젝트 전체를 push한 뒤 [Streamlit Community Cloud](https://share.streamlit.io/)에서
저장소와 `app.py`를 선택해 배포합니다. API 키나 별도 환경변수는 필요하지 않습니다.

## 데이터 출처

- [환경공간정보서비스 토지피복지도 맵 서비스](https://aid.mcee.go.kr/api/land.do)
- [서울열린데이터광장 스마트서울 도시데이터 센서 환경정보](https://data.seoul.go.kr/dataList/OA-22833/A/1/datasetView.do)

두 자료 모두 공공누리 제1유형 출처표시 조건에 따라 출처를 표시했습니다. S-DoT 측정값은
연구·탐색용 데이터이며 통신지연, 장애 또는 현장 여건에 따른 결측·이상값이 있을 수 있습니다.
