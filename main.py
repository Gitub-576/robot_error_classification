# -*- coding: utf-8 -*-
"""
산업용 로봇 오류 유형 분류 실습실
실행: streamlit run main.py
"""

import io
import re
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# ---------------------------------------------------------
# 1. 기본 설정 및 데이터 불러오기
# ---------------------------------------------------------
st.set_page_config(
    page_title="산업용 로봇 오류 유형 분류 실습실",
    page_icon="🤖",
    layout="wide"
)

UCI_LP1_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/robotfailure-mld/lp1.data"
)

CHANNELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

# 원본 파일의 실제 순서: 각 시점마다 Fx, Fy, Fz, Tx, Ty, Tz
CANONICAL_FEATURES = [
    f"{channel}{time}"
    for time in range(1, 16)
    for channel in CHANNELS
]


def make_default_columns():
    return ["Class"] + CANONICAL_FEATURES


def is_number(value):
    try:
        float(str(value).strip())
        return True
    except (ValueError, TypeError):
        return False





def read_robot_data(file_bytes):
    """
    UCI LP1 원본 데이터 파싱
    데이터 1개 = Class 1개 + 센서값 90개
    센서값은 6개씩 총 15줄로 구성
    """

    decoded = None

    # 파일 인코딩 확인
    for encoding in ["utf-8-sig", "cp949", "latin1"]:
        try:
            decoded = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded is None:
        raise ValueError("파일 인코딩을 읽을 수 없습니다.")

    rows = []
    current_class = None
    current_values = []

    for line_number, line in enumerate(decoded.splitlines(), 1):

        line = line.strip()

        # 빈 줄과 주석 제외
        if not line or line.startswith("#"):
            continue

        # 공백, 탭, 쉼표, 세미콜론으로 분리
        values = re.split(r"[\s,;]+", line)
        values = [v.strip() for v in values if v.strip()]

        # -------------------------------------------------
        # 형식 1: 한 줄에 Class + 센서값 90개
        # -------------------------------------------------
        if len(values) == 91:

            if current_class is not None:
                raise ValueError(
                    f"{line_number}번째 줄: 이전 데이터가 완성되지 않았습니다."
                )

            if not all(is_number(v) for v in values[1:]):
                raise ValueError(
                    f"{line_number}번째 줄: 센서값에 숫자가 아닌 값이 있습니다."
                )

            rows.append(
                [values[0]] + [float(v) for v in values[1:]]
            )

            continue

        # -------------------------------------------------
        # 형식 2: Class + 센서값 6개가 한 줄에 있는 경우
        # -------------------------------------------------
        if len(values) == 7 and not is_number(values[0]):

            if current_class is not None:
                raise ValueError(
                    f"{line_number}번째 줄: 이전 Class의 센서값이 "
                    "90개보다 적습니다."
                )

            if not all(is_number(v) for v in values[1:]):
                raise ValueError(
                    f"{line_number}번째 줄: 센서값이 숫자가 아닙니다."
                )

            current_class = values[0]
            current_values = [float(v) for v in values[1:]]

        # -------------------------------------------------
        # 형식 3: Class만 있는 한 줄
        # -------------------------------------------------
        elif len(values) == 1 and not is_number(values[0]):

            if current_class is not None:
                raise ValueError(
                    f"{line_number}번째 줄: 이전 Class의 센서값이 "
                    "90개보다 적습니다."
                )

            current_class = values[0]
            current_values = []

        # -------------------------------------------------
        # 형식 4: 센서 측정값 6개
        # -------------------------------------------------
        elif len(values) == 6:

            if current_class is None:
                raise ValueError(
                    f"{line_number}번째 줄: Class 정보가 없습니다."
                )

            if not all(is_number(v) for v in values):
                raise ValueError(
                    f"{line_number}번째 줄: 숫자가 아닌 센서값이 있습니다."
                )

            current_values.extend(
                [float(v) for v in values]
            )

        else:
            raise ValueError(
                f"{line_number}번째 줄의 형식이 올바르지 않습니다. "
                f"분리된 값의 개수: {len(values)}개"
            )

        # -------------------------------------------------
        # 센서값 90개가 모이면 한 행 완성
        # -------------------------------------------------
        if current_class is not None and len(current_values) == 90:

            rows.append(
                [current_class] + current_values
            )

            current_class = None
            current_values = []

        elif len(current_values) > 90:
            raise ValueError(
                f"{line_number}번째 줄: 센서값이 90개를 초과했습니다."
            )

    # -------------------------------------------------
    # 마지막 데이터 검증
    # -------------------------------------------------
    if current_class is not None:
        raise ValueError(
            f"Class '{current_class}'의 센서값이 "
            f"90개가 아닙니다. 현재 {len(current_values)}개입니다."
        )

    if not rows:
        raise ValueError("읽을 수 있는 데이터가 없습니다.")

    # -------------------------------------------------
    # 91개 컬럼 생성
    # -------------------------------------------------
    columns = ["Class"] + [
        f"{channel}{i}"
        for i in range(1, 16)
        for channel in CHANNELS
    ]

    parsed = pd.DataFrame(
        rows,
        columns=columns
    )

    # 센서값 숫자형 확인
    for col in columns[1:]:
        parsed[col] = pd.to_numeric(parsed[col], errors="raise")

    return parsed, columns[1:]





@st.cache_data(show_spinner=False)
def download_uci_lp1():
    response = requests.get(UCI_LP1_URL, timeout=20)
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------
# 2. 화면 상단
# ---------------------------------------------------------
st.title("🤖 산업용 로봇 오류 유형 분류 실습실")
st.caption(
    "힘·토크 센서 데이터로 로봇의 정상 작동 및 오류 유형을 분류하는 "
    "머신러닝 프로젝트"
)

st.markdown(
    """
    이 프로그램은 UCI Machine Learning Repository의 **Robot Execution Failures**
    데이터 중 LP1을 사용합니다. 15개 시점에서 측정한 6종류의 힘·토크 값,
    총 90개 변수를 이용하여 `Class`를 예측합니다.
    """
)

st.divider()

# ---------------------------------------------------------
# 3. 데이터 선택 및 업로드
# ---------------------------------------------------------
st.header("① 프로젝트 소개")
intro1, intro2, intro3 = st.columns(3)
intro1.metric("독립 변수", "90개", "힘·토크 측정값")
intro2.metric("측정 시점", "15개", "각 시점마다 6개 측정")
intro3.metric("종속 변수", "Class", "오류 유형 레이블")

st.markdown(
    """
    **분류 원리**

    로봇이 작업을 수행하는 동안 센서에는 각 방향의 힘(Force)과 토크(Torque)가
    기록됩니다. 오류 상황에 따라 센서값의 크기와 변화 양상이 달라질 수 있습니다.
    머신러닝 모델은 이러한 측정 패턴과 정답 Class 사이의 관계를 학습한 뒤,
    새로운 측정값이 주어졌을 때 가장 알맞은 Class를 예측합니다.

    - 힘: `Fx`, `Fy`, `Fz` — X, Y, Z축 방향의 힘
    - 토크: `Tx`, `Ty`, `Tz` — X, Y, Z축 기준 회전 효과
    - 시간 번호 1~15: 오류 감지 직후부터 기록한 15개 측정 시점
    """
)

st.markdown(
    "[UCI 데이터셋 공식 페이지](https://archive.ics.uci.edu/dataset/138/robot+execution+failures) "
    " · [LP1 원본 데이터 설명](https://archive.ics.uci.edu/ml/machine-learning-databases/robotfailure-mld/robotfailure.html)"
)

st.divider()
st.header("② 데이터 확인")

source_choice = st.radio(
    "데이터 불러오기 방법",
    ["UCI에서 LP1 원본 데이터 불러오기", "내 컴퓨터에서 데이터 파일 업로드"],
    horizontal=True
)

uploaded_file = None
raw_bytes = None

if source_choice == "UCI에서 LP1 원본 데이터 불러오기":
    if st.button("UCI LP1 데이터 불러오기", type="primary"):
        try:
            raw_bytes = download_uci_lp1()
            st.session_state["robot_lp1_bytes"] = raw_bytes
            st.success("UCI LP1 데이터를 불러왔습니다.")
        except Exception as exc:
            st.error(
                "UCI 서버에서 데이터를 가져오지 못했습니다. "
                "인터넷 연결을 확인하거나 원본 파일을 직접 업로드해 주세요."
            )
            st.caption(f"오류 내용: {exc}")

    if "robot_lp1_bytes" in st.session_state:
        raw_bytes = st.session_state["robot_lp1_bytes"]
    else:
        st.info("위 버튼을 눌러 UCI의 LP1 원본 데이터를 불러오세요.")

else:
    uploaded_file = st.file_uploader(
        "LP1 데이터 파일 선택 (.data, .csv, .txt)",
        type=["data", "csv", "txt"]
    )
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()

if raw_bytes is None:
    st.stop()

try:
    df, feature_cols = read_robot_data(raw_bytes)
except Exception as exc:
    st.error(f"데이터 형식 오류: {exc}")
    st.info(
        "LP1 파일은 각 행에 Class 1개와 숫자형 센서 측정값 90개가 있어야 합니다. "
        "원본 파일을 다시 확인해 주세요."
    )
    st.stop()

classes = sorted(df["Class"].dropna().unique().tolist(), key=str)
class_counts = df["Class"].value_counts()

# 데이터 요약 카드
m1, m2, m3, m4 = st.columns(4)
m1.metric("전체 데이터 수", f"{len(df):,}개")
m2.metric("전체 열 수", f"{df.shape[1]:,}개")
m3.metric("분류 클래스 수", f"{df['Class'].nunique()}개")
m4.metric("결측 센서값", f"{int(df[feature_cols].isna().sum().sum()):,}개")

st.subheader("데이터 상위 5개 행")
st.dataframe(df.head(), use_container_width=True)

with st.expander("컬럼 이름과 자료형 확인"):
    info_df = pd.DataFrame({
        "컬럼 이름": df.columns,
        "자료형": [str(df[c].dtype) for c in df.columns],
        "결측치 수": [int(df[c].isna().sum()) for c in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Class별 데이터 개수")
    st.dataframe(
        class_counts.rename_axis("Class").reset_index(name="데이터 수"),
        use_container_width=True
    )
with c2:
    st.subheader("데이터 품질 확인")
    st.write(f"- 중복 행 수: **{int(df.duplicated().sum())}개**")
    st.write(f"- Class 결측치: **{int(df['Class'].isna().sum())}개**")
    st.write(f"- 센서값 결측치: **{int(df[feature_cols].isna().sum().sum())}개**")
    all_missing = [c for c in feature_cols if df[c].isna().all()]
    if all_missing:
        st.error(f"모든 값이 결측인 센서 컬럼이 있습니다: {all_missing}")
    else:
        st.success("모든 센서 컬럼에 적어도 하나 이상의 숫자 측정값이 있습니다.")

st.divider()

# ---------------------------------------------------------
# 4. 탐색 및 시각화
# ---------------------------------------------------------
st.header("③ 데이터 탐색 및 시각화")

# 실제 컬럼명에서 센서 축을 찾는다. UCI 헤더 없는 원본은 위에서 표준 이름 생성.
channel_columns = {}
for channel in CHANNELS:
    cols = [
        c for c in feature_cols
        if re.fullmatch(rf"{channel}\d+", str(c), flags=re.IGNORECASE)
    ]
    channel_columns[channel] = cols

# 헤더가 비표준인 경우에는 원본의 정해진 순서를 사용해 축별로 묶는다.
if any(len(channel_columns[ch]) != 15 for ch in CHANNELS):
    channel_columns = {
        channel: [feature_cols[i] for i in range(channel_index, 90, 6)]
        for channel_index, channel in enumerate(CHANNELS)
    }

axis_names = {"Fx": "X축 힘", "Fy": "Y축 힘", "Fz": "Z축 힘",
              "Tx": "X축 토크", "Ty": "Y축 토크", "Tz": "Z축 토크"}

# 시각화용 데이터 구성: 행별 평균 절댓값은 클래스별 센서 크기 비교에 활용
plot_rows = []
for _, row in df.iterrows():
    for channel in CHANNELS:
        values = pd.to_numeric(row[channel_columns[channel]], errors="coerce")
        for value in values:
            if pd.notna(value):
                plot_rows.append({
                    "Class": str(row["Class"]),
                    "축": axis_names[channel],
                    "종류": "힘" if channel.startswith("F") else "토크",
                    "측정값": float(value)
                })
plot_df = pd.DataFrame(plot_rows)

# 그래프 1: Class별 데이터 개수
fig1 = px.bar(
    class_counts.rename_axis("Class").reset_index(name="데이터 수"),
    x="Class", y="데이터 수", color="Class",
    title="Class별 데이터 개수",
    text="데이터 수"
)
fig1.update_layout(showlegend=False, xaxis_title="Class", yaxis_title="데이터 개수")
st.plotly_chart(fig1, use_container_width=True)
st.caption(
    f"전체 {len(df)}개 중 가장 많은 Class는 "
    f"{class_counts.idxmax()}({class_counts.max()}개)이며, "
    f"가장 적은 Class는 {class_counts.idxmin()}({class_counts.min()}개)입니다. "
    "클래스별 표본 수 차이는 모델 평가에 영향을 줄 수 있습니다."
)

# 그래프 2: 힘 분포
force_df = plot_df[plot_df["종류"] == "힘"]
fig2 = px.box(
    force_df, x="축", y="측정값", color="축",
    title="X·Y·Z축 힘 측정값 분포",
    points="outliers"
)
fig2.update_layout(showlegend=False, xaxis_title="힘 측정 축", yaxis_title="힘 측정값")
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "상자그림의 중앙선은 중앙값, 상자 범위는 가운데 50%의 데이터 분포를 나타냅니다. "
    "축마다 측정값의 범위와 이상치가 어떻게 다른지 비교할 수 있습니다."
)

# 그래프 3: 토크 분포
torque_df = plot_df[plot_df["종류"] == "토크"]
fig3 = px.box(
    torque_df, x="축", y="측정값", color="축",
    title="X·Y·Z축 토크 측정값 분포",
    points="outliers"
)
fig3.update_layout(showlegend=False, xaxis_title="토크 측정 축", yaxis_title="토크 측정값")
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "토크의 축별 중앙값과 분포 폭을 비교할 수 있습니다. "
    "일부 큰 값은 특정 시점의 회전 효과가 상대적으로 컸음을 보여줄 수 있습니다."
)

# 행별 축 평균 절댓값을 계산해 클래스 간 크기 비교
def make_instance_summary(kind):
    selected_channels = ["Fx", "Fy", "Fz"] if kind == "힘" else ["Tx", "Ty", "Tz"]
    records = []
    for row_index, row in df.iterrows():
        for channel in selected_channels:
            vals = pd.to_numeric(row[channel_columns[channel]], errors="coerce")
            records.append({
                "Class": str(row["Class"]),
                "축": axis_names[channel],
                "평균 절댓값": float(vals.abs().mean()) if vals.notna().any() else np.nan,
                "데이터 번호": row_index
            })
    return pd.DataFrame(records)

force_summary = make_instance_summary("힘")
torque_summary = make_instance_summary("토크")

fig4 = px.box(
    force_summary, x="Class", y="평균 절댓값", color="축",
    title="Class별 힘 측정값 비교 (각 행의 축별 평균 절댓값)"
)
fig4.update_layout(xaxis_title="Class", yaxis_title="평균 절댓값")
st.plotly_chart(fig4, use_container_width=True)
st.caption(
    "각 데이터 행에서 15개 시점의 힘 절댓값 평균을 계산해 비교한 그래프입니다. "
    "Class에 따라 힘의 크기 분포가 달라지는지 살펴볼 수 있습니다."
)

fig5 = px.box(
    torque_summary, x="Class", y="평균 절댓값", color="축",
    title="Class별 토크 측정값 비교 (각 행의 축별 평균 절댓값)"
)
fig5.update_layout(xaxis_title="Class", yaxis_title="평균 절댓값")
st.plotly_chart(fig5, use_container_width=True)
st.caption(
    "각 데이터 행에서 15개 시점의 토크 절댓값 평균을 계산한 그래프입니다. "
    "클래스 사이의 분포가 겹치는 정도를 통해 구분의 어려움을 탐색할 수 있습니다."
)

st.divider()

# ---------------------------------------------------------
# 5. 전처리 및 모델 학습
# ---------------------------------------------------------
st.header("④ 데이터 전처리 및 모델 학습")

st.markdown(
    """
    - 중복 행은 모델 학습 전에 제거합니다.
    - Class가 없는 행은 학습할 수 없으므로 제외합니다.
    - 센서값의 결측치는 학습 데이터의 중앙값으로 대체합니다.
    - 중앙값 대체 과정은 `Pipeline` 안에서 수행하므로 테스트 데이터의 정보가
      전처리 학습에 섞이지 않습니다.
    - 학습/테스트 데이터는 기본 8:2로 나누고, 각 Class 비율을 유지하도록
      층화 분할(Stratified Split)을 적용합니다.
    """
)

test_ratio = st.slider(
    "테스트 데이터 비율",
    min_value=0.1, max_value=0.4, value=0.2, step=0.05,
    help="기본값 0.2는 학습 데이터 80%, 테스트 데이터 20%를 의미합니다."
)

# 학습용 데이터 준비
model_df = df.drop_duplicates().copy()
model_df = model_df.dropna(subset=["Class"])
X = model_df[feature_cols].copy()
y = model_df["Class"].astype(str).copy()

# 전부 결측인 특성은 중앙값을 계산할 수 없으므로 안내 후 중단
empty_features = [c for c in feature_cols if X[c].isna().all()]

st.write(f"- 원본 행 수: **{len(df)}개**")
st.write(f"- 중복 제거 후 행 수: **{len(model_df)}개**")
st.write(f"- 제거된 중복 행 수: **{len(df) - len(model_df)}개**")
st.write(f"- 모델 입력 변수 수: **{X.shape[1]}개**")
st.write(f"- 결측 센서값은 학습 데이터 중앙값으로 대체합니다.")

if empty_features:
    st.error(
        "모든 값이 결측인 센서 컬럼이 있어 모델을 학습할 수 없습니다: "
        + ", ".join(map(str, empty_features))
    )
    st.stop()

if y.nunique() < 2:
    st.error("분류 모델을 학습하려면 최소 2개 이상의 Class가 필요합니다.")
    st.stop()

class_min_count = int(y.value_counts().min())
n_samples = len(y)
n_classes = y.nunique()
n_test = int(np.ceil(n_samples * test_ratio))
n_train = n_samples - n_test

if class_min_count < 2:
    st.error(
        "가장 적은 Class의 데이터가 2개 미만이어서 층화 분할을 할 수 없습니다. "
        "각 Class에 최소 2개 이상의 데이터가 필요합니다."
    )
    st.stop()

if n_test < n_classes or n_train < n_classes:
    st.error(
        "현재 테스트 비율에서는 학습 또는 테스트 데이터에 모든 Class를 "
        "포함할 수 없습니다. 테스트 비율을 조정하거나 데이터를 더 확보해 주세요."
    )
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        random_state=42,
        stratify=y
    )
except ValueError as exc:
    st.error(f"층화 데이터 분할에 실패했습니다: {exc}")
    st.stop()

st.success(
    f"층화 분할 완료: 학습 {len(X_train)}개 / 테스트 {len(X_test)}개"
)

# 두 모델 모두 동일한 전처리 파이프라인을 사용
models = {
    "의사결정나무": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "랜덤 포레스트": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
}

st.info(
    "**랜덤 포레스트**는 여러 의사결정나무를 학습시킨 뒤 각 나무의 예측을 "
    "종합하여 최종 클래스를 결정하는 앙상블 분류 모델입니다."
)

# ---------------------------------------------------------
# 6. 성능 평가
# ---------------------------------------------------------
st.header("⑤ 모델 성능 평가")

results = {}
reports = {}
predictions = {}
cv_results = {}

with st.spinner("두 분류 모델을 학습하고 테스트 데이터로 평가하고 있습니다..."):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0
        )
        p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0
        )

        results[model_name] = {
            "정확도": accuracy,
            "Macro 정밀도": p_macro,
            "Macro 재현율": r_macro,
            "Macro F1": f_macro,
            "Weighted 정밀도": p_weight,
            "Weighted 재현율": r_weight,
            "Weighted F1": f_weight
        }
        predictions[model_name] = pred
        reports[model_name] = classification_report(
            y_test, pred, labels=classes,
            output_dict=True, zero_division=0
        )

        # 교차 검증은 학습 데이터 안에서만 수행. 각 fold에서 Pipeline이 다시 학습됨.
        train_min_class = int(y_train.value_counts().min())
        if train_min_class >= 2:
            folds = min(5, train_min_class)
            cv = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=42
            )
            try:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring="f1_macro", n_jobs=-1
                )
                cv_results[model_name] = {
                    "가능": True,
                    "folds": folds,
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores))
                }
            except Exception as exc:
                cv_results[model_name] = {
                    "가능": False,
                    "reason": str(exc)
                }
        else:
            cv_results[model_name] = {
                "가능": False,
                "reason": "학습 데이터에서 가장 적은 Class의 표본이 2개 미만입니다."
            }

# 성능 비교 표
comparison = pd.DataFrame(results).T
st.subheader("두 모델의 전체 성능 비교")
st.dataframe(
    comparison.style.format("{:.3f}"),
    use_container_width=True
)

metric_cols = st.columns(4)
for col, metric in zip(
    metric_cols,
    ["정확도", "Macro 정밀도", "Macro 재현율", "Macro F1"]
):
    col.metric(
        metric,
        f"{comparison.loc['랜덤 포레스트', metric]:.3f}",
        f"의사결정나무 {comparison.loc['의사결정나무', metric]:.3f}"
    )

# 클래스별 성능
st.subheader("Class별 성능")
selected_model_name = st.selectbox(
    "클래스별 상세 성능을 확인할 모델",
    list(models.keys())
)
selected_report = reports[selected_model_name]

class_report_rows = []
for class_name in classes:
    row = selected_report.get(str(class_name), {})
    class_report_rows.append({
        "Class": str(class_name),
        "정밀도": row.get("precision", 0),
        "재현율": row.get("recall", 0),
        "F1-score": row.get("f1-score", 0),
        "표본 수": int(row.get("support", 0))
    })

st.dataframe(
    pd.DataFrame(class_report_rows).style.format({
        "정밀도": "{:.3f}", "재현율": "{:.3f}", "F1-score": "{:.3f}"
    }),
    use_container_width=True
)

# 혼동 행렬
st.subheader("혼동 행렬")
cm = confusion_matrix(
    y_test, predictions[selected_model_name], labels=classes
)
fig_cm = px.imshow(
    cm,
    x=[str(c) for c in classes],
    y=[str(c) for c in classes],
    text_auto=True,
    color_continuous_scale="Blues",
    title=f"{selected_model_name} 혼동 행렬"
)
fig_cm.update_layout(
    xaxis_title="모델 예측 Class",
    yaxis_title="실제 Class"
)
st.plotly_chart(fig_cm, use_container_width=True)
st.caption(
    "혼동 행렬의 행은 실제 Class, 열은 모델이 예측한 Class입니다. "
    "대각선 숫자가 클수록 해당 Class를 올바르게 분류한 데이터가 많습니다. "
    "대각선 밖의 값은 다른 Class로 잘못 분류한 사례입니다."
)

# 교차 검증 결과
st.subheader("Stratified K-Fold 교차 검증")
for name, cv_result in cv_results.items():
    if cv_result["가능"]:
        st.write(
            f"- **{name}**: {cv_result['folds']}개 fold의 Macro F1 평균 "
            f"**{cv_result['mean']:.3f}**, 표준편차 **{cv_result['std']:.3f}**"
        )
    else:
        st.warning(f"{name}: 교차 검증을 수행하지 못했습니다. {cv_result['reason']}")

# ---------------------------------------------------------
# 7. 실제 결과를 바탕으로 자동 해석
# ---------------------------------------------------------
st.subheader("성능 평가 결과 해석")

for model_name in models:
    result = results[model_name]
    st.markdown(f"**{model_name}**")
    st.write(
        f"- 정확도: 테스트 데이터 {len(y_test)}개 중 "
        f"{accuracy_score(y_test, predictions[model_name]) * len(y_test):.0f}개를 "
        f"올바르게 분류했으며, 정확도는 **{result['정확도']:.3f} "
        f"({result['정확도'] * 100:.1f}%)**입니다."
    )
    st.write(
        f"- 정밀도: Macro 정밀도는 **{result['Macro 정밀도']:.3f}**입니다. "
        "모델이 특정 Class라고 예측한 결과가 실제로 해당 Class인 정도를 "
        "클래스별로 동일한 비중으로 평균한 값입니다."
    )
    st.write(
        f"- 재현율: Macro 재현율은 **{result['Macro 재현율']:.3f}**입니다. "
        "실제 각 Class 데이터를 모델이 얼마나 찾아냈는지 나타냅니다."
    )
    st.write(
        f"- F1-score: Macro F1은 **{result['Macro F1']:.3f}**입니다. "
        "정밀도와 재현율의 균형을 나타내며 각 Class를 동일한 비중으로 평가합니다."
    )

    report = reports[model_name]
    per_class_f1 = {
        str(c): report.get(str(c), {}).get("f1-score", 0)
        for c in classes
    }
    hardest_class = min(per_class_f1, key=per_class_f1.get)
    easiest_class = max(per_class_f1, key=per_class_f1.get)
    st.write(
        f"- Class별 F1-score를 비교하면 **{easiest_class}**의 값이 가장 높고 "
        f"**{hardest_class}**의 값이 가장 낮습니다. 따라서 현재 테스트 데이터에서는 "
        f"{hardest_class}를 상대적으로 구분하기 어려웠습니다."
    )

# 혼동이 가장 많은 Class 쌍 찾기
best_name_for_analysis = max(
    results,
    key=lambda name: (results[name]["Macro F1"], results[name]["정확도"])
)
best_pred = predictions[best_name_for_analysis]
best_cm = confusion_matrix(y_test, best_pred, labels=classes)
off_diag = best_cm.copy()
np.fill_diagonal(off_diag, 0)

if off_diag.max() > 0:
    i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    confusion_text = (
        f"가장 많이 혼동한 사례는 실제 Class **{classes[i]}**를 "
        f"Class **{classes[j]}**로 예측한 경우로, **{off_diag[i, j]}개**입니다."
    )
else:
    confusion_text = "테스트 데이터에서 서로 다른 Class 사이의 오분류가 관찰되지 않았습니다."

st.markdown("**혼동 행렬 분석**")
st.write(confusion_text)

# Macro F1 우선, 동률이면 정확도 비교
tree_score = results["의사결정나무"]["Macro F1"]
forest_score = results["랜덤 포레스트"]["Macro F1"]

if forest_score > tree_score:
    better_model_name = "랜덤 포레스트"
    other_model_name = "의사결정나무"
elif tree_score > forest_score:
    better_model_name = "의사결정나무"
    other_model_name = "랜덤 포레스트"
else:
    if results["랜덤 포레스트"]["정확도"] >= results["의사결정나무"]["정확도"]:
        better_model_name = "랜덤 포레스트"
        other_model_name = "의사결정나무"
    else:
        better_model_name = "의사결정나무"
        other_model_name = "랜덤 포레스트"

st.subheader("모델 평가를 통해 도출된 결론")
st.write(
    f"1. **상대적으로 우수한 모델:** 테스트 데이터의 Macro F1-score를 우선 비교한 결과, "
    f"현재 데이터 분할에서는 **{better_model_name}**이 더 높은 성능을 보였습니다. "
    f"(Macro F1: {results[better_model_name]['Macro F1']:.3f}, "
    f"{other_model_name}: {results[other_model_name]['Macro F1']:.3f})"
)
st.write(
    f"2. **오류 유형 구분 능력:** {better_model_name}의 정확도는 "
    f"{results[better_model_name]['정확도'] * 100:.1f}%이고 Macro F1-score는 "
    f"{results[better_model_name]['Macro F1']:.3f}입니다. 이 결과는 이번에 분리한 "
    "테스트 데이터에 대한 결과이며, 실제 산업 현장에서도 같은 성능을 보장하지 않습니다."
)

best_report = reports[better_model_name]
best_class_f1 = {
    str(c): best_report.get(str(c), {}).get("f1-score", 0)
    for c in classes
}
hardest = min(best_class_f1, key=best_class_f1.get)
st.write(
    f"3. **분류가 어려운 유형:** 현재 테스트 데이터에서는 Class {hardest}의 "
    f"F1-score가 가장 낮아 상대적으로 분류가 어려웠습니다. "
    "클래스별 표본 수와 혼동 행렬을 함께 확인할 필요가 있습니다."
)

smallest_class = class_counts.idxmin()
st.write(
    f"4. **데이터 양과 불균형의 영향:** 원본 LP1은 총 {len(df)}개로 데이터 규모가 "
    f"크지 않고, 가장 적은 Class는 {smallest_class}({class_counts.min()}개)입니다. "
    "표본이 적은 클래스는 학습 과정에서 다양한 패턴을 충분히 반영하지 못할 수 있습니다."
)
st.write(
    "5. **한계와 개선점:** 데이터가 88개로 비교적 적고 특정 작업 상황(LP1)에 한정되어 "
    "있다는 점이 한계입니다. 다양한 로봇과 작업 환경의 데이터를 추가 수집하고, "
    "시계열 특징을 요약하거나 다른 분류 모델과 비교하면 개선 가능성을 탐색할 수 있습니다."
)

# 최종 모델 저장: Macro F1 우선 선택
final_model = models[better_model_name]
st.session_state["robot_final_model"] = final_model
st.session_state["robot_feature_cols"] = feature_cols
st.session_state["robot_class_values"] = classes
st.session_state["robot_final_model_name"] = better_model_name

st.success(
    f"새로운 센서값 분류에는 **{better_model_name}** 모델을 사용합니다. "
    "선택 기준은 테스트 데이터의 Macro F1-score입니다."
)

st.divider()

# ---------------------------------------------------------
# 8. 새로운 센서 데이터 예측
# ---------------------------------------------------------
st.header("⑥ 로봇 오류 유형 분류")

st.markdown(
    """
    아래에 새로운 로봇 센서 측정값 90개를 입력하면, 위에서 학습한 최종 모델이
    Class를 예측합니다. 실제 센서 데이터의 측정 단위와 범위에 맞는 값을 입력해야 합니다.
    기본값 0은 입력 예시일 뿐 실제 측정값을 의미하지 않습니다.
    """
)

with st.form("prediction_form"):
    st.subheader("힘(Force) 측정값")
    input_values = {}

    for channel in ["Fx", "Fy", "Fz"]:
        with st.expander(f"{axis_names[channel]}: {channel}1 ~ {channel}15", expanded=False):
            cols = st.columns(5)
            for time_index in range(1, 16):
                with cols[(time_index - 1) % 5]:
                    input_values[f"{channel}{time_index}"] = st.number_input(
                        f"{channel}{time_index}",
                        value=0.0,
                        step=0.1,
                        key=f"input_{channel}_{time_index}"
                    )

    st.subheader("토크(Torque) 측정값")
    for channel in ["Tx", "Ty", "Tz"]:
        with st.expander(f"{axis_names[channel]}: {channel}1 ~ {channel}15", expanded=False):
            cols = st.columns(5)
            for time_index in range(1, 16):
                with cols[(time_index - 1) % 5]:
                    input_values[f"{channel}{time_index}"] = st.number_input(
                        f"{channel}{time_index}",
                        value=0.0,
                        step=0.1,
                        key=f"input_{channel}_{time_index}"
                    )

    submitted = st.form_submit_button("입력한 센서값으로 Class 분류하기", type="primary")

if submitted:
    # 데이터 컬럼 순서와 같은 순서로 입력값 구성
    ordered_input = {}
    for col in feature_cols:
        match = re.fullmatch(r"(Fx|Fy|Fz|Tx|Ty|Tz)(\d+)", str(col), flags=re.IGNORECASE)
        if match:
            key = match.group(1) + match.group(2)
            if key not in input_values:
                st.error(f"입력 화면에서 {key} 측정값을 찾을 수 없습니다.")
                st.stop()
            ordered_input[col] = input_values[key]
        else:
            st.error(
                f"컬럼 이름 '{col}'을 Fx1~Tz15 형식으로 해석할 수 없습니다. "
                "예측 입력 기능은 표준 LP1 센서 컬럼 이름을 필요로 합니다."
            )
            st.stop()

    input_df = pd.DataFrame([ordered_input], columns=feature_cols)
    predicted_class = final_model.predict(input_df)[0]

    st.success("분류가 완료되었습니다.")
    st.markdown("### 예측 결과")
    st.markdown(
        f"<div style='padding:22px; border-radius:12px; "
        f"background-color:#e8f1ff; text-align:center;'>"
        f"<div style='font-size:18px; color:#334155;'>예측된 Class</div>"
        f"<div style='font-size:48px; font-weight:700; color:#1d4ed8;'>"
        f"{predicted_class}</div></div>",
        unsafe_allow_html=True
    )
    st.info(
        "이 데이터셋의 Class는 원본 레이블을 그대로 표시합니다. "
        "확인되지 않은 숫자 레이블의 의미를 임의로 붙이지 않았습니다. "
        "Class의 구체적인 의미는 데이터셋의 공식 설명 및 수업 자료에서 확인해 주세요."
    )

st.divider()
st.caption(
    "데이터 출처: Lopes, L. & Camarinha-Matos, L. (1998). "
    "Robot Execution Failures. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5M89N"
)
