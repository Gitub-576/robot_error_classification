
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import re
from io import StringIO

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="산업용 로봇 오류 유형 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 산업용 로봇 오류 유형 예측 실습실")
st.caption("기계공학 × 로봇·자동화 × 머신러닝 분류 프로젝트")

st.markdown("""
산업용 로봇에서 측정된 힘과 토크 데이터를 이용하여
로봇의 정상 작동 및 오류 유형을 분류하는 프로그램입니다.

**주의:** 이 프로그램은 오류 감지 이후 수집된 센서 데이터를 분류합니다.
오류 발생 전의 미래 상태를 예측하는 프로그램은 아닙니다.
""")

st.divider()

# =========================================================
# 2. 데이터 관련 설정
# =========================================================

UCI_URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/robotfailure-mld/lp1.data"
)

CHANNELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

# 원본 파일의 측정 순서:
# 1번째 시점의 Fx, Fy, Fz, Tx, Ty, Tz
# 2번째 시점의 Fx, Fy, Fz, Tx, Ty, Tz ...
FEATURES = [
    f"{channel}{time}"
    for time in range(1, 16)
    for channel in CHANNELS
]

EXPECTED_COLUMNS = ["Class"] + FEATURES

FORCE_CHANNELS = ["Fx", "Fy", "Fz"]
TORQUE_CHANNELS = ["Tx", "Ty", "Tz"]


# =========================================================
# 3. 데이터 읽기 및 형식 확인
# =========================================================

def read_robot_data(file_object):
    """
    공백, 탭, 쉼표, 세미콜론으로 구분된 파일을 읽는다.
    헤더가 있는 파일과 헤더가 없는 원본 파일을 모두 지원한다.
    """

    if hasattr(file_object, "read"):
        raw_text = file_object.read()

        if isinstance(raw_text, bytes):
            raw_text = raw_text.decode("utf-8-sig", errors="replace")
    else:
        raw_text = str(file_object)

    if not raw_text.strip():
        raise ValueError("파일에 데이터가 없습니다.")

    # 구분자에 관계없이 공백 단위로 데이터를 분리한다.
    raw_df = pd.read_csv(
        StringIO(raw_text),
        sep=r"[\s,;]+",
        engine="python",
        header=None,
        comment="#"
    )

    raw_df = raw_df.dropna(how="all")

    if raw_df.empty:
        raise ValueError("읽을 수 있는 데이터가 없습니다.")

    raw_df = raw_df.reset_index(drop=True)

    # 첫 번째 행에 헤더가 있는지 확인한다.
    first_row = [
        str(value).strip().replace("\ufeff", "")
        for value in raw_df.iloc[0].tolist()
    ]

    first_cell = first_row[0].lower()

    has_header = (
        first_cell in ["class", "label", "target"]
        or any(
            re.fullmatch(
                r"(?:f[xz y]?|t[xz y]?)[1-9][0-5]?",
                value,
                re.IGNORECASE
            )
            for value in first_row[1:]
        )
    )

    if has_header:
        raw_df.columns = first_row
        raw_df = raw_df.iloc[1:].reset_index(drop=True)

        # 컬럼 이름 정리
        cleaned_columns = []

        for column in raw_df.columns:
            name = str(column).strip().replace("\ufeff", "")

            if name.lower() in ["class", "label", "target"]:
                cleaned_columns.append("Class")
            else:
                cleaned_columns.append(name)

        raw_df.columns = cleaned_columns

        # 컬럼 이름이 실제 예상 구조와 일치하는지 확인
        if len(raw_df.columns) != 91:
            raise ValueError(
                f"헤더가 있는 데이터의 열 개수가 91개가 아닙니다. "
                f"현재 {len(raw_df.columns)}개입니다."
            )

        if len(set(raw_df.columns)) != 91:
            raise ValueError("중복된 컬럼 이름이 있습니다.")

        expected_set = set(EXPECTED_COLUMNS)

        if set(raw_df.columns) != expected_set:
            missing = expected_set - set(raw_df.columns)
            extra = set(raw_df.columns) - expected_set

            raise ValueError(
                f"컬럼 구성이 예상과 다릅니다.\n"
                f"누락된 컬럼: {list(missing)[:10]}\n"
                f"추가된 컬럼: {list(extra)[:10]}"
            )

        # 컬럼 순서를 원본 기준으로 통일
        raw_df = raw_df[EXPECTED_COLUMNS]

    else:
        # 헤더가 없는 원본 데이터
        if raw_df.shape[1] != 91:
            raise ValueError(
                f"원본 데이터는 Class 1개와 센서 데이터 90개, "
                f"총 91개 열이어야 합니다. 현재 {raw_df.shape[1]}개입니다."
            )

        raw_df.columns = EXPECTED_COLUMNS

    # Class를 제외한 센서 데이터는 숫자로 변환
    for column in FEATURES:
        raw_df[column] = pd.to_numeric(
            raw_df[column],
            errors="coerce"
        )

    # 센서 데이터가 숫자가 아닌 경우 확인
    invalid_count = raw_df[FEATURES].isna().sum().sum()

    if invalid_count > 0:
        raise ValueError(
            f"센서 측정값에 숫자가 아닌 값 또는 결측치가 "
            f"총 {invalid_count}개 있습니다. 파일의 구분자와 내용을 확인해 주세요."
        )

    # Class 결측치 확인
    if raw_df["Class"].isna().any():
        raise ValueError("Class에 비어 있는 값이 있습니다.")

    raw_df["Class"] = raw_df["Class"].astype(str).str.strip()

    if (raw_df["Class"] == "").any():
        raise ValueError("Class에 빈 문자열이 있습니다.")

    if raw_df.empty:
        raise ValueError("유효한 데이터가 없습니다.")

    return raw_df


@st.cache_data(show_spinner=False)
def download_uci_data():
    response = requests.get(UCI_URL, timeout=20)
    response.raise_for_status()
    response.encoding = "utf-8"
    return read_robot_data(response.text)


# =========================================================
# 4. 사이드바: 데이터 불러오기
# =========================================================

st.sidebar.header("📂 데이터 설정")

data_option = st.sidebar.radio(
    "데이터 불러오기 방식",
    ["UCI 원본 데이터 사용", "파일 직접 업로드"]
)

df = None

if data_option == "UCI 원본 데이터 사용":

    if st.sidebar.button("UCI 데이터 불러오기", type="primary"):
        try:
            st.session_state["robot_df"] = download_uci_data()
        except Exception as e:
            st.sidebar.error(
                f"데이터를 불러오지 못했습니다.\n{e}"
            )

    if "robot_df" in st.session_state:
        df = st.session_state["robot_df"]
    else:
        st.info(
            "왼쪽 사이드바에서 'UCI 데이터 불러오기'를 눌러 주세요."
        )

else:
    uploaded_file = st.sidebar.file_uploader(
        "LP1 데이터 파일 선택",
        type=["data", "txt", "csv"]
    )

    if uploaded_file is not None:
        try:
            df = read_robot_data(uploaded_file)
            st.session_state["robot_df"] = df
        except Exception as e:
            st.error(f"데이터 파일을 읽는 중 오류가 발생했습니다.\n\n{e}")

if df is None:
    st.markdown("### 프로젝트를 시작해 보세요!")
    st.info(
        "데이터를 불러오면 프로젝트 소개, 데이터 확인, "
        "시각화, 모델 학습 및 예측 기능을 사용할 수 있습니다."
    )

    st.markdown(
        f"[UCI Robot Execution Failures 데이터셋 바로가기]({UCI_URL})"
    )
    st.stop()


# =========================================================
# 5. 데이터 기본 정보
# =========================================================

df = df.copy()

# Class별 데이터 개수
class_counts = df["Class"].value_counts()

classes = sorted(
    df["Class"].unique().tolist(),
    key=lambda x: str(x)
)

# 중복 데이터 개수
duplicate_count = int(df.duplicated().sum())

# 결측치 개수
missing_count = int(df.isna().sum().sum())


# =========================================================
# 6. 메뉴 구성
# =========================================================

menu = st.sidebar.radio(
    "📌 메뉴 선택",
    [
        "① 프로젝트 소개",
        "② 데이터 확인",
        "③ 데이터 탐색 및 시각화",
        "④ 데이터 전처리 및 모델 학습",
        "⑤ 모델 성능 평가",
        "⑥ 로봇 오류 유형 예측"
    ]
)


# =========================================================
# 7. 프로젝트 소개
# =========================================================

if menu == "① 프로젝트 소개":

    st.header("① 프로젝트 소개")

    st.subheader("프로젝트 주제")
    st.markdown("### 산업용 로봇의 작동 오류 유형 예측")

    st.write("""
    산업용 로봇은 물체를 집거나 이동시키는 과정에서
    충돌이나 장애물 등의 문제를 경험할 수 있습니다.

    이러한 상황에서 로봇의 센서가 측정하는 힘과 토크 데이터는
    작동 상태를 파악하는 중요한 정보가 됩니다.

    본 프로젝트에서는 힘과 토크 데이터를 머신러닝 모델에 학습시켜
    로봇의 작동 상태 및 오류 유형을 분류합니다.
    """)

    st.subheader("프로젝트 목표")

    st.markdown("""
    1. 로봇의 힘과 토크 측정 데이터를 수집한다.
    2. 데이터를 탐색하고 특징을 분석한다.
    3. 머신러닝 분류 모델을 학습한다.
    4. 두 모델의 성능을 비교하고 평가한다.
    5. 새로운 센서 측정값으로 오류 유형을 분류한다.
    """)

    st.subheader("데이터 출처")

    st.markdown("""
    - 제공 기관: UCI Machine Learning Repository
    - 데이터셋: Robot Execution Failures
    - 사용 데이터: LP1
    - 데이터 유형: 시계열 센서 데이터
    - 관측 구간: 315ms
    """)

    st.markdown(
        "[UCI 데이터셋 확인하기](https://archive.ics.uci.edu/dataset/138/robot+execution+failures)"
    )

    st.subheader("독립 변수와 종속 변수")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 독립 변수 (X)")
        st.info("""
        총 90개의 센서 측정값

        - Fx: X축 방향 힘
        - Fy: Y축 방향 힘
        - Fz: Z축 방향 힘
        - Tx: X축 방향 토크
        - Ty: Y축 방향 토크
        - Tz: Z축 방향 토크

        각 센서를 15개 시점에서 측정합니다.
        """)

    with col2:
        st.markdown("#### 종속 변수 (Y)")
        st.success("""
        Class

        로봇의 정상 작동 또는 오류 유형을
        나타내는 정답 레이블입니다.

        모델은 90개의 센서 측정값을 이용하여
        Class를 분류합니다.
        """)

    st.subheader("머신러닝 분류 원리")

    st.write("""
    각 데이터 행에는 로봇의 힘과 토크 측정값 90개가 기록되어 있습니다.

    머신러닝 모델은 이 측정값과 실제 Class 사이의 관계를 학습합니다.

    학습이 끝나면 새로운 센서 측정값을 입력했을 때
    학습한 패턴과 비교하여 가장 가능성이 높은 Class를 출력합니다.
    """)

    st.subheader("사용 모델")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**의사결정나무 (Decision Tree)**")
        st.write("""
        여러 조건을 차례대로 나누면서
        데이터를 분류하는 모델입니다.
        """)

    with col2:
        st.markdown("**랜덤 포레스트 (Random Forest)**")
        st.write("""
        여러 의사결정나무를 학습시키고
        그 결과를 종합하여 분류하는 모델입니다.
        """)

    st.divider()

    st.caption(
        f"현재 불러온 데이터: {len(df)}개 행 / "
        f"{len(FEATURES)}개 독립 변수 / {len(classes)}개 Class"
    )


# =========================================================
# 8. 데이터 확인
# =========================================================

elif menu == "② 데이터 확인":

    st.header("② 데이터 확인")

    st.subheader("데이터 기본 정보")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("전체 데이터 수", f"{len(df):,}개")
    c2.metric("전체 컬럼 수", f"{df.shape[1]}개")
    c3.metric("결측치", f"{missing_count}개")
    c4.metric("중복 데이터", f"{duplicate_count}개")

    st.divider()

    st.subheader("데이터 상위 5개")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    st.subheader("컬럼 이름과 자료형")

    info_df = pd.DataFrame({
        "컬럼 이름": df.columns,
        "자료형": [str(df[col].dtype) for col in df.columns],
        "결측치 수": [int(df[col].isna().sum()) for col in df.columns]
    })

    st.dataframe(info_df, use_container_width=True)

    st.divider()

    st.subheader("Class별 데이터 개수")

    class_df = class_counts.reset_index()
    class_df.columns = ["Class", "데이터 개수"]

    st.dataframe(class_df, use_container_width=True)

    fig = px.bar(
        class_df,
        x="Class",
        y="데이터 개수",
        color="Class",
        text="데이터 개수",
        title="Class별 데이터 개수"
    )

    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="데이터 개수",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    if class_counts.nunique() > 1:
        st.warning(
            "Class마다 데이터 개수가 다릅니다. "
            "클래스 불균형이 모델 성능에 영향을 줄 수 있습니다."
        )
    else:
        st.success("모든 Class의 데이터 개수가 같습니다.")


# =========================================================
# 9. 데이터 탐색 및 시각화
# =========================================================

elif menu == "③ 데이터 탐색 및 시각화":

    st.header("③ 데이터 탐색 및 시각화")

    st.write("""
    센서 측정값의 분포와 Class별 차이를 시각적으로 확인합니다.
    """)

    # 1. Class별 개수
    st.subheader("1. Class별 데이터 개수")

    class_df = class_counts.reset_index()
    class_df.columns = ["Class", "개수"]

    fig1 = px.bar(
        class_df,
        x="Class",
        y="개수",
        color="Class",
        text="개수",
        title="Class별 데이터 분포"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.caption(
        f"전체 {len(df)}개 데이터가 {len(classes)}개의 Class로 구성되어 있습니다. "
        "막대 높이를 비교하여 클래스별 데이터 불균형 여부를 확인할 수 있습니다."
    )

    st.divider()

    # 센서 데이터를 긴 형태로 변환
    def make_long_data(data, channels):
        result = []

        for channel in channels:
            cols = [f"{channel}{i}" for i in range(1, 16)]

            temp = data[["Class"] + cols].copy()

            long = temp.melt(
                id_vars="Class",
                value_vars=cols,
                var_name="측정 시점",
                value_name="측정값"
            )

            long["센서 축"] = channel
            long["측정 시점"] = long["측정 시점"].str.extract(
                r"(\d+)"
            ).astype(int)

            result.append(long)

        return pd.concat(result, ignore_index=True)

    # 2. 힘 데이터 분포
    st.subheader("2. X축·Y축·Z축 힘 데이터 분포")

    force_long = make_long_data(df, FORCE_CHANNELS)

    fig2 = px.box(
        force_long,
        x="센서 축",
        y="측정값",
        color="센서 축",
        points="outliers",
        title="힘 센서 측정값 분포"
    )

    fig2.update_layout(
        xaxis_title="힘의 방향",
        yaxis_title="힘 측정값"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "각 축의 힘 측정값이 어느 범위에 분포하는지 확인할 수 있습니다. "
        "상자와 수염의 범위가 넓을수록 측정값의 변동이 큰 편입니다."
    )

    st.divider()

    # 3. 토크 데이터 분포
    st.subheader("3. X축·Y축·Z축 토크 데이터 분포")

    torque_long = make_long_data(df, TORQUE_CHANNELS)

    fig3 = px.box(
        torque_long,
        x="센서 축",
        y="측정값",
        color="센서 축",
        points="outliers",
        title="토크 센서 측정값 분포"
    )

    fig3.update_layout(
        xaxis_title="토크의 방향",
        yaxis_title="토크 측정값"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "각 방향의 토크 측정값의 중앙값과 분포 범위를 비교할 수 있습니다."
    )

    st.divider()

    # 각 행의 15개 측정값 평균을 계산
    def make_axis_average(data, channels, kind):
        result = []

        for channel in channels:
            cols = [f"{channel}{i}" for i in range(1, 16)]

            temp = pd.DataFrame({
                "Class": data["Class"],
                "측정값 평균": data[cols].mean(axis=1),
                "센서 축": channel
            })

            result.append(temp)

        return pd.concat(result, ignore_index=True)

    # 4. Class별 힘 비교
    st.subheader("4. Class별 힘 측정값 비교")

    force_avg = make_axis_average(
        df, FORCE_CHANNELS, "힘"
    )

    fig4 = px.box(
        force_avg,
        x="Class",
        y="측정값 평균",
        color="Class",
        facet_col="센서 축",
        title="Class별 힘 측정값 평균 비교"
    )

    fig4.update_layout(
        showlegend=False,
        xaxis_title="Class",
        yaxis_title="15개 시점의 힘 평균"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.caption(
        "각 Class에서 X·Y·Z축 힘의 평균이 어떻게 다른지 비교합니다. "
        "Class별 분포 차이가 크다면 오류 유형을 구분하는 특징이 될 가능성이 있습니다."
    )

    st.divider()

    # 5. Class별 토크 비교
    st.subheader("5. Class별 토크 측정값 비교")

    torque_avg = make_axis_average(
        df, TORQUE_CHANNELS, "토크"
    )

    fig5 = px.box(
        torque_avg,
        x="Class",
        y="측정값 평균",
        color="Class",
        facet_col="센서 축",
        title="Class별 토크 측정값 평균 비교"
    )

    fig5.update_layout(
        showlegend=False,
        xaxis_title="Class",
        yaxis_title="15개 시점의 토크 평균"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.caption(
        "각 Class의 X·Y·Z축 토크 평균 분포를 비교합니다. "
        "서로 다른 오류 유형 사이에서 토크 특성에 차이가 있는지 확인할 수 있습니다."
    )

    st.divider()

    st.info(
        "그래프는 데이터의 경향을 탐색하기 위한 것입니다. "
        "분포가 다르다는 사실만으로 특정 센서값이 오류의 원인이라고 "
        "단정할 수는 없습니다."
    )


# =========================================================
# 10. 전처리 및 모델 학습
# =========================================================

elif menu == "④ 데이터 전처리 및 모델 학습":

    st.header("④ 데이터 전처리 및 모델 학습")

    st.subheader("전처리 과정")

    st.markdown("""
    1. Class와 센서 데이터를 분리한다.
    2. 완전히 동일한 중복 데이터를 제거한다.
    3. 학습 데이터와 테스트 데이터를 8:2로 분리한다.
    4. 학습 데이터의 결측치 처리 기준을 학습한다.
    5. 의사결정나무와 랜덤 포레스트를 학습한다.

    **데이터 누수 방지:** 결측치 대체 기준은 학습 데이터에서만 계산합니다.
    테스트 데이터는 모델 평가를 위해 분리해 둡니다.
    """)

    st.write("### 데이터 현황")

    c1, c2, c3 = st.columns(3)

    c1.metric("원본 행 수", len(df))
    c2.metric("중복 행 수", duplicate_count)
    c3.metric("사용 가능한 변수", len(FEATURES))

    # 중복 데이터 제거
    clean_df = df.drop_duplicates().copy()

    if len(clean_df) < len(df):
        st.warning(
            f"중복 데이터 {len(df) - len(clean_df)}개를 제거했습니다."
        )

    # 독립 변수와 종속 변수 분리
    X = clean_df[FEATURES].copy()
    y = clean_df["Class"].copy()

    st.write("### 독립 변수와 종속 변수")

    st.write(f"- 독립 변수 X: {X.shape[1]}개")
    st.write(f"- 종속 변수 Y: Class")
    st.write(f"- 분석 데이터: {len(clean_df)}개")

    class_counts_clean = y.value_counts()

    st.write("### 전처리 후 Class별 개수")

    st.dataframe(
        class_counts_clean.rename("개수").reset_index(),
        use_container_width=True
    )

    # 층화 분할 가능 여부 확인
    min_class_count = int(class_counts_clean.min())

    if len(classes) < 2:
        st.error("분류를 위해서는 최소 2개 이상의 Class가 필요합니다.")
        st.stop()

    if min_class_count < 2:
        st.error(
            "특정 Class의 데이터가 2개 미만이므로 "
            "층화 학습·테스트 분할을 진행할 수 없습니다."
        )
        st.stop()

    # 데이터 분할 비율 설정
    test_ratio = st.slider(
        "테스트 데이터 비율",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05
    )

    random_state = 42

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_ratio,
            random_state=random_state,
            stratify=y
        )

    except ValueError as e:
        st.error(
            f"층화 데이터 분할에 실패했습니다.\n\n{e}\n\n"
            "테스트 비율을 조정하거나 데이터를 추가해 주세요."
        )
        st.stop()

    st.success("층화 데이터 분할이 완료되었습니다.")

    c1, c2 = st.columns(2)

    c1.metric("학습 데이터", f"{len(X_train)}개")
    c2.metric("테스트 데이터", f"{len(X_test)}개")

    st.caption(
        "층화 분할을 사용하여 학습 데이터와 테스트 데이터에 "
        "Class별 비율이 최대한 비슷하게 유지되도록 했습니다."
    )

    st.divider()

    st.subheader("모델 학습")

    st.write("""
    두 모델 모두 동일한 학습 데이터와 테스트 데이터를 사용합니다.

    - 의사결정나무: 조건에 따라 데이터를 나누는 방식
    - 랜덤 포레스트: 여러 의사결정나무의 결과를 종합하는 방식
    """)

    # 전처리와 모델을 하나의 Pipeline으로 구성
    # SimpleImputer는 학습 데이터의 중앙값만 학습한다.
    models = {
        "의사결정나무": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                random_state=random_state,
                class_weight="balanced"
            ))
        ]),

        "랜덤 포레스트": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])
    }

    if st.button("🚀 두 모델 학습 및 평가 시작", type="primary"):

        result = {}

        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, pred)

            precision_macro, recall_macro, f1_macro, _ = (
                precision_recall_fscore_support(
                    y_test,
                    pred,
                    average="macro",
                    zero_division=0
                )
            )

            precision_weighted, recall_weighted, f1_weighted, _ = (
                precision_recall_fscore_support(
                    y_test,
                    pred,
                    average="weighted",
                    zero_division=0
                )
            )

            report = classification_report(
                y_test,
                pred,
                labels=classes,
                output_dict=True,
                zero_division=0
            )

            matrix = confusion_matrix(
                y_test,
                pred,
                labels=classes
            )

            result[name] = {
                "model": model,
                "prediction": pred,
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1_weighted": f1_weighted,
                "report": report,
                "matrix": matrix
            }

            progress.progress((i + 1) / 2)

        # Macro F1을 우선으로 최종 모델 선택
        # 동점이면 정확도를 비교한다.
        best_name = max(
            result.keys(),
            key=lambda name: (
                result[name]["f1_macro"],
                result[name]["accuracy"]
            )
        )

        best_model = result[best_name]["model"]

        # 성능 평가와 별도로 전체 데이터로 최종 예측 모델 학습
        final_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", (
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    class_weight="balanced",
                    n_jobs=-1
                )
                if best_name == "랜덤 포레스트"
                else DecisionTreeClassifier(
                    random_state=random_state,
                    class_weight="balanced"
                )
            ))
        ])

        final_model.fit(X, y)

        # 교차 검증
        cv_result = None
        cv_reason = None

        min_train_class = int(y_train.value_counts().min())

        # 전체 데이터에 대한 교차 검증 가능 조건
        if min_class_count >= 2:

            n_splits = min(5, min_class_count)

            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )

            try:
                scores = cross_val_score(
                    best_model,
                    X,
                    y,
                    cv=cv,
                    scoring="f1_macro",
                    n_jobs=1,
                    error_score="raise"
                )

                cv_result = {
                    "folds": n_splits,
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "scores": scores.tolist()
                }

            except Exception as e:
                cv_reason = str(e)

        else:
            cv_reason = "일부 Class의 데이터가 부족합니다."

        # 세션에 결과 저장
        st.session_state["model_results"] = result
        st.session_state["best_name"] = best_name
        st.session_state["final_model"] = final_model
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["classes"] = classes
        st.session_state["clean_df"] = clean_df
        st.session_state["cv_result"] = cv_result
        st.session_state["cv_reason"] = cv_reason
        st.session_state["train_size"] = len(X_train)
        st.session_state["test_size"] = len(X_test)
        st.session_state["split_ratio"] = test_ratio

        st.success("두 모델의 학습 및 평가가 완료되었습니다!")

        st.balloons()

    # 이전에 학습한 결과가 있다면 표시
    if "model_results" in st.session_state:

        st.divider()
        st.subheader("학습 결과 요약")

        result = st.session_state["model_results"]

        comparison = []

        for name, r in result.items():

            comparison.append({
                "모델": name,
                "정확도": r["accuracy"],
                "Macro F1-score": r["f1_macro"],
                "Weighted F1-score": r["f1_weighted"]
            })

        comparison_df = pd.DataFrame(comparison)

        st.dataframe(
            comparison_df.style.format({
                "정확도": "{:.4f}",
                "Macro F1-score": "{:.4f}",
                "Weighted F1-score": "{:.4f}"
            }),
            use_container_width=True
        )

        st.success(
            f"현재 선택된 최종 모델: {st.session_state['best_name']}"
        )

        st.info(
            "상세한 성능 평가 및 자동 결과 해석은 "
            "왼쪽 메뉴의 '⑤ 모델 성능 평가'에서 확인할 수 있습니다."
        )


# =========================================================
# 11. 모델 성능 평가
# =========================================================

elif menu == "⑤ 모델 성능 평가":

    st.header("⑤ 모델 성능 평가")

    if "model_results" not in st.session_state:
        st.warning(
            "먼저 '④ 데이터 전처리 및 모델 학습'에서 "
            "모델을 학습해 주세요."
        )
        st.stop()

    result = st.session_state["model_results"]
    best_name = st.session_state["best_name"]

    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    cv_result = st.session_state["cv_result"]

    st.success(f"최종 선택 모델: {best_name}")

    st.caption(
        f"실제 테스트 데이터 {len(X_test)}개를 사용한 평가 결과입니다."
    )

    # 두 모델 비교
    st.subheader("1. 두 모델의 성능 비교")

    comparison = []

    for name, r in result.items():

        comparison.append({
            "모델": name,
            "정확도": r["accuracy"],
            "Macro 정밀도": r["precision_macro"],
            "Macro 재현율": r["recall_macro"],
            "Macro F1": r["f1_macro"],
            "Weighted F1": r["f1_weighted"]
        })

    comparison_df = pd.DataFrame(comparison)

    st.dataframe(
        comparison_df.style.format({
            col: "{:.4f}"
            for col in comparison_df.columns
            if col != "모델"
        }),
        use_container_width=True
    )

    fig_compare = px.bar(
        comparison_df.melt(
            id_vars="모델",
            var_name="평가 지표",
            value_name="점수"
        ),
        x="평가 지표",
        y="점수",
        color="모델",
        barmode="group",
        title="모델별 성능 지표 비교",
        range_y=[0, 1]
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    st.divider()

    # 모델 선택
    selected_name = st.selectbox(
        "상세 평가를 확인할 모델",
        list(result.keys()),
        index=list(result.keys()).index(best_name)
    )

    r = result[selected_name]

    # 핵심 지표
    st.subheader("2. 주요 성능 평가 지표")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("정확도", f"{r['accuracy']:.4f}")
    c2.metric("Macro 정밀도", f"{r['precision_macro']:.4f}")
    c3.metric("Macro 재현율", f"{r['recall_macro']:.4f}")
    c4.metric("Macro F1-score", f"{r['f1_macro']:.4f}")

    st.write("### 전체 성능")

    overall_df = pd.DataFrame([
        {
            "평균 방식": "Macro 평균",
            "정밀도": r["precision_macro"],
            "재현율": r["recall_macro"],
            "F1-score": r["f1_macro"]
        },
        {
            "평균 방식": "Weighted 평균",
            "정밀도": r["precision_weighted"],
            "재현율": r["recall_weighted"],
            "F1-score": r["f1_weighted"]
        }
    ])

    st.dataframe(
        overall_df.style.format({
            "정밀도": "{:.4f}",
            "재현율": "{:.4f}",
            "F1-score": "{:.4f}"
        }),
        use_container_width=True
    )

    st.caption("""
    Macro 평균은 모든 Class의 성능을 동일한 비중으로 평균한 값입니다.
    Weighted 평균은 각 Class의 데이터 개수를 고려하여 평균한 값입니다.
    따라서 클래스 불균형이 있을 때 두 결과를 함께 확인하는 것이 중요합니다.
    """)

    st.divider()

    # 클래스별 평가
    st.subheader("3. Class별 성능")

    report = r["report"]

    per_class = []

    for class_name in classes:

        item = report[str(class_name)]

        per_class.append({
            "Class": class_name,
            "정밀도": item["precision"],
            "재현율": item["recall"],
            "F1-score": item["f1-score"],
            "실제 데이터 수": int(item["support"])
        })

    per_class_df = pd.DataFrame(per_class)

    st.dataframe(
        per_class_df.style.format({
            "정밀도": "{:.4f}",
            "재현율": "{:.4f}",
            "F1-score": "{:.4f}"
        }),
        use_container_width=True
    )

    fig_class = px.bar(
        per_class_df.melt(
            id_vars="Class",
            value_vars=["정밀도", "재현율", "F1-score"],
            var_name="평가 지표",
            value_name="점수"
        ),
        x="Class",
        y="점수",
        color="평가 지표",
        barmode="group",
        title="Class별 성능 비교",
        range_y=[0, 1]
    )

    st.plotly_chart(fig_class, use_container_width=True)

    st.divider()

    # 혼동 행렬
    st.subheader("4. 혼동 행렬 (Confusion Matrix)")

    matrix = r["matrix"]

    fig_matrix = px.imshow(
        matrix,
        x=[str(c) for c in classes],
        y=[str(c) for c in classes],
        text_auto=True,
        color_continuous_scale="Blues",
        labels={
            "x": "예측한 Class",
            "y": "실제 Class",
            "color": "데이터 개수"
        },
        title="실제 Class와 예측 Class 비교"
    )

    st.plotly_chart(fig_matrix, use_container_width=True)

    st.caption("""
    혼동 행렬의 세로축은 실제 Class, 가로축은 모델이 예측한 Class입니다.
    대각선에 있는 숫자가 클수록 올바르게 분류한 데이터가 많다는 의미입니다.
    대각선 바깥의 숫자는 다른 Class로 잘못 분류한 데이터입니다.
    """)

    # 오분류 분석
    errors = []

    for i, actual in enumerate(classes):

        for j, predicted in enumerate(classes):

            if i != j and matrix[i, j] > 0:

                errors.append({
                    "실제 Class": actual,
                    "잘못 예측한 Class": predicted,
                    "오분류 개수": int(matrix[i, j])
                })

    if errors:

        error_df = pd.DataFrame(errors).sort_values(
            "오분류 개수",
            ascending=False
        )

        st.write("### 주요 오분류 유형")

        st.dataframe(
            error_df,
            use_container_width=True
        )

        most_error = error_df.iloc[0]

        st.warning(
            f"가장 많이 발생한 오분류는 실제 Class "
            f"'{most_error['실제 Class']}'를 "
            f"'{most_error['잘못 예측한 Class']}'로 "
            f"분류한 경우이며, {most_error['오분류 개수']}개입니다."
        )

    else:
        st.success("테스트 데이터에서 오분류가 발생하지 않았습니다.")

    st.divider()

    # 교차 검증
    st.subheader("5. Stratified K-Fold 교차 검증")

    if cv_result is not None:

        c1, c2 = st.columns(2)

        c1.metric(
            "교차 검증 평균 Macro F1",
            f"{cv_result['mean']:.4f}"
        )

        c2.metric(
            "표준편차",
            f"{cv_result['std']:.4f}"
        )

        st.write(
            f"교차 검증 횟수: {cv_result['folds']}회"
        )

        cv_df = pd.DataFrame({
            "검증 회차": [
                f"{i + 1}회차"
                for i in range(cv_result["folds"])
            ],
            "Macro F1-score": cv_result["scores"]
        })

        fig_cv = px.bar(
            cv_df,
            x="검증 회차",
            y="Macro F1-score",
            text="Macro F1-score",
            title="교차 검증 회차별 성능"
        )

        st.plotly_chart(fig_cv, use_container_width=True)

        st.caption(
            "평균은 교차 검증의 전반적인 성능을 나타내며, "
            "표준편차가 작을수록 각 검증 과정의 성능 차이가 작은 편입니다."
        )

    else:
        st.warning(
            "교차 검증을 수행할 수 없었습니다. "
            + str(st.session_state["cv_reason"])
        )

    st.divider()

    # =====================================================
    # 12. 실제 지표를 이용한 자동 해석
    # =====================================================

    st.subheader("6. 성능 평가 결과 해석")

    acc = r["accuracy"]
    precision = r["precision_macro"]
    recall = r["recall_macro"]
    f1 = r["f1_macro"]
    weighted_f1 = r["f1_weighted"]

    # 1. 정확도 해석
    st.markdown("#### ① 정확도 (Accuracy)")

    st.write(
        f"정확도는 전체 테스트 데이터 중 올바르게 분류한 "
        f"데이터의 비율입니다. 이번 모델의 정확도는 "
        f"{acc:.4f}({acc * 100:.2f}%)입니다."
    )

    st.write(
        f"테스트 데이터 {len(y_test)}개 중 "
        f"{int(round(acc * len(y_test)))}개를 올바르게 분류했습니다."
    )

    # 2. 정밀도 및 재현율
    st.markdown("#### ② 정밀도와 재현율")

    st.write(
        f"Macro 정밀도는 {precision:.4f}입니다. "
        "이는 각 Class에 대해 해당 Class라고 예측한 결과가 "
        "얼마나 정확했는지를 동일한 비중으로 평균한 값입니다."
    )

    st.write(
        f"Macro 재현율은 {recall:.4f}입니다. "
        "이는 실제 각 Class의 데이터를 얼마나 놓치지 않고 "
        "찾아냈는지를 동일한 비중으로 평균한 값입니다."
    )

    if precision > recall:
        st.info(
            "정밀도가 재현율보다 높습니다. "
            "예측한 Class의 정확성은 상대적으로 높지만, "
            "실제 Class 중 일부를 놓치는 경우가 있을 수 있습니다."
        )
    elif recall > precision:
        st.info(
            "재현율이 정밀도보다 높습니다. "
            "실제 Class를 찾아내는 비율은 상대적으로 높지만, "
            "다른 Class를 해당 Class로 잘못 예측하는 경우가 있을 수 있습니다."
        )
    else:
        st.info("Macro 정밀도와 재현율이 같습니다.")

    # 3. F1 해석
    st.markdown("#### ③ F1-score")

    st.write(
        f"Macro F1-score는 {f1:.4f}, "
        f"Weighted F1-score는 {weighted_f1:.4f}입니다."
    )

    st.write(
        "F1-score는 정밀도와 재현율의 조화평균으로, "
        "두 지표를 함께 고려하여 분류 성능을 평가합니다."
    )

    if f1 >= 0.8:
        st.write("이번 모델은 테스트 데이터에서 비교적 높은 Macro F1-score를 보였습니다.")
    elif f1 >= 0.6:
        st.write("이번 모델은 일정 수준의 분류 성능을 보였지만 개선할 여지가 있습니다.")
    else:
        st.write("이번 모델의 Macro F1-score가 낮은 편이므로 분류 성능 개선이 필요합니다.")

    if abs(f1 - weighted_f1) >= 0.1:
        st.warning(
            "Macro F1-score와 Weighted F1-score 사이에 차이가 있습니다. "
            "클래스별 데이터 수 차이가 전체 평가 결과에 영향을 주었을 가능성이 있습니다."
        )

    # 4. 혼동 행렬 해석
    st.markdown("#### ④ 혼동 행렬 분석")

    recalls = per_class_df.set_index("Class")["재현율"]

    easiest = recalls.idxmax()
    hardest = recalls.idxmin()

    st.write(
        f"가장 높은 재현율을 기록한 Class는 '{easiest}'이며 "
        f"재현율은 {recalls.max():.4f}입니다."
    )

    st.write(
        f"가장 낮은 재현율을 기록한 Class는 '{hardest}'이며 "
        f"재현율은 {recalls.min():.4f}입니다."
    )

    st.write(
        "따라서 이번 모델에서는 "
        f"'{easiest}'를 상대적으로 잘 분류했으며, "
        f"'{hardest}'의 실제 데이터를 찾아내는 데 어려움이 있었습니다."
    )

    # 5. 두 모델 비교
    st.markdown("#### ⑤ 두 모델 비교 및 평가")

    tree_f1 = result["의사결정나무"]["f1_macro"]
    forest_f1 = result["랜덤 포레스트"]["f1_macro"]

    tree_acc = result["의사결정나무"]["accuracy"]
    forest_acc = result["랜덤 포레스트"]["accuracy"]

    tree_report = result["의사결정나무"]["report"]
    forest_report = result["랜덤 포레스트"]["report"]

    tree_class_recalls = [
        tree_report[str(c)]["recall"]
        for c in classes
    ]

    forest_class_recalls = [
        forest_report[str(c)]["recall"]
        for c in classes
    ]

    tree_min_recall = min(tree_class_recalls)
    forest_min_recall = min(forest_class_recalls)

    st.write(
        f"- 의사결정나무 Macro F1: {tree_f1:.4f}, "
        f"정확도: {tree_acc:.4f}"
    )

    st.write(
        f"- 랜덤 포레스트 Macro F1: {forest_f1:.4f}, "
        f"정확도: {forest_acc:.4f}"
    )

    if forest_f1 > tree_f1:
        st.write(
            "랜덤 포레스트가 Macro F1-score에서 더 높은 성능을 보였습니다."
        )
    elif tree_f1 > forest_f1:
        st.write(
            "의사결정나무가 Macro F1-score에서 더 높은 성능을 보였습니다."
        )
    else:
        st.write(
            "두 모델의 Macro F1-score가 같습니다."
        )

    if forest_min_recall > tree_min_recall:
        st.write(
            "랜덤 포레스트는 가장 낮은 클래스별 재현율 측면에서도 "
            "상대적으로 좋은 결과를 보였습니다."
        )
    elif tree_min_recall > forest_min_recall:
        st.write(
            "의사결정나무는 가장 낮은 클래스별 재현율 측면에서 "
            "상대적으로 좋은 결과를 보였습니다."
        )
    else:
        st.write("두 모델의 최저 클래스별 재현율은 같습니다.")

    st.divider()

    # =====================================================
    # 13. 프로젝트 최종 결론 자동 작성
    # =====================================================

    st.subheader("7. 모델 평가를 통해 도출된 결론")

    st.markdown("#### ① 어떤 모델이 더 우수했는가?")

    st.write(
        f"두 모델 중 Macro F1-score를 기준으로 선택한 모델은 "
        f"'{best_name}'입니다. "
        f"해당 모델의 Macro F1-score는 "
        f"{result[best_name]['f1_macro']:.4f}입니다."
    )

    st.markdown("#### ② 오류 유형을 어느 정도 구분할 수 있었는가?")

    st.write(
        f"선택 모델의 정확도는 {acc:.4f}, "
        f"Macro F1-score는 {f1:.4f}입니다. "
        "이 결과를 통해 테스트 데이터에서 로봇의 작동 상태를 "
        "어느 정도 분류할 수 있는지 확인했습니다."
    )

    st.markdown("#### ③ 상대적으로 분류하기 어려운 오류 유형")

    st.write(
        f"'{hardest}'의 재현율이 {recalls.min():.4f}로 "
        "가장 낮았습니다. 따라서 이 Class를 다른 Class와 "
        "구분하는 데 어려움이 있었던 것으로 해석할 수 있습니다."
    )

    st.markdown("#### ④ 데이터 수와 클래스 불균형의 영향")

    largest_class = class_counts_clean.idxmax()
    smallest_class = class_counts_clean.idxmin()

    st.write(
        f"데이터가 가장 많은 Class는 '{largest_class}' "
        f"({int(class_counts_clean.max())}개), "
        f"가장 적은 Class는 '{smallest_class}' "
        f"({int(class_counts_clean.min())}개)입니다."
    )

    st.write("""
    특정 Class의 데이터가 적으면 모델이 해당 유형의 다양한
    센서 패턴을 충분히 학습하지 못할 가능성이 있습니다.
    또한 전체 데이터 자체가 많지 않으므로 테스트 데이터의
    구성에 따라 평가 점수가 달라질 수 있습니다.
    """)

    st.markdown("#### ⑤ 프로젝트의 한계와 개선점")

    st.write("""
    - LP1 데이터는 총 88개로 데이터의 양이 제한적입니다.
    - 특정 Class의 데이터가 다른 Class보다 적을 수 있습니다.
    - 학습 데이터와 테스트 데이터의 분할 방식에 따라 성능이 달라질 수 있습니다.
    - 90개 센서값을 그대로 사용하는 방식이 모든 시간적 특징을 충분히 설명하지 못할 수 있습니다.
    - 실제 산업용 로봇에 적용하기 위해서는 다양한 작업 환경과 센서 조건에서 추가 검증이 필요합니다.

    향후에는 더 많은 데이터를 수집하고, 시간에 따른 힘과 토크의 변화량,
    평균, 최댓값 등의 특징을 추가하여 모델의 성능을 개선할 수 있습니다.
    """)

    st.info(
        "이번 결과는 제한된 LP1 데이터에 대한 실험 결과이며, "
        "실제 산업 현장에서의 오류 진단 성능을 보장하지 않습니다."
    )


# =========================================================
# 14. 새로운 센서 데이터로 Class 예측
# =========================================================

elif menu == "⑥ 로봇 오류 유형 예측":

    st.header("⑥ 로봇 오류 유형 예측")

    if "final_model" not in st.session_state:
        st.warning(
            "먼저 '④ 데이터 전처리 및 모델 학습'에서 "
            "두 모델을 학습해 주세요."
        )
        st.stop()

    final_model = st.session_state["final_model"]
    best_name = st.session_state["best_name"]
    clean_df = st.session_state["clean_df"]

    st.success(f"현재 사용 중인 예측 모델: {best_name}")

    st.write("""
    새로운 로봇 센서 측정값 90개를 입력하면
    학습된 모델이 해당 데이터의 Class를 분류합니다.

    입력값은 15개 측정 시점의 Fx, Fy, Fz, Tx, Ty, Tz 값으로 구성됩니다.
    """)

    # 기본 입력값은 실제 학습 데이터의 각 컬럼 중앙값으로 설정
    default_values = clean_df[FEATURES].median()

    st.info(
        "입력 편의를 위해 각 입력란에는 실제 학습 데이터의 중앙값을 "
        "기본값으로 표시합니다. 실제 센서 측정값이 있다면 해당 값으로 수정하세요."
    )

    with st.form("prediction_form"):

        st.subheader("힘 센서 입력값")

        force_input = {}

        for channel in FORCE_CHANNELS:

            with st.expander(f"{channel} : {'X' if channel == 'Fx' else 'Y' if channel == 'Fy' else 'Z'}축 힘", expanded=False):

                cols = st.columns(5)

                for i in range(1, 16):

                    name = f"{channel}{i}"

                    with cols[(i - 1) % 5]:

                        force_input[name] = st.number_input(
                            name,
                            value=float(default_values[name]),
                            key=f"force_{name}"
                        )

        st.divider()

        st.subheader("토크 센서 입력값")

        torque_input = {}

        for channel in TORQUE_CHANNELS:

            with st.expander(f"{channel} : {'X' if channel == 'Tx' else 'Y' if channel == 'Ty' else 'Z'}축 토크", expanded=False):

                cols = st.columns(5)

                for i in range(1, 16):

                    name = f"{channel}{i}"

                    with cols[(i - 1) % 5]:

                        torque_input[name] = st.number_input(
                            name,
                            value=float(default_values[name]),
                            key=f"torque_{name}"
                        )

        st.divider()

        predict_button = st.form_submit_button(
            "🔍 로봇 오류 유형 분류하기",
            type="primary",
            use_container_width=True
        )

    if predict_button:

        # 입력값 90개를 원래 모델의 컬럼 순서로 정리
        input_values = {
            **force_input,
            **torque_input
        }

        input_df = pd.DataFrame(
            [[input_values[col] for col in FEATURES]],
            columns=FEATURES
        )

        # 예측 실행
        prediction = final_model.predict(input_df)[0]

        st.divider()

        st.subheader("예측 결과")

        st.markdown(
            f"""
            <div style="
                background-color: #e8f1ff;
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid #b5d0f5;
            ">
                <p style="font-size:18px; color:#315b91;">
                    머신러닝 분류 결과
                </p>
                <h1 style="color:#174a8b; font-size:42px;">
                    {prediction}
                </h1>
                <p style="font-size:16px; color:#315b91;">
                    예측된 Class
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### 클래스별 예측 확률")

        if hasattr(final_model, "predict_proba"):

            probabilities = final_model.predict_proba(input_df)[0]

            model_classes = final_model.named_steps["model"].classes_

            probability_df = pd.DataFrame({
                "Class": model_classes.astype(str),
                "예측 확률": probabilities
            })

            probability_df = probability_df.sort_values(
                "예측 확률",
                ascending=False
            )

            fig_probability = px.bar(
                probability_df,
                x="Class",
                y="예측 확률",
                color="Class",
                text=probability_df["예측 확률"].map(
                    lambda x: f"{x:.1%}"
                ),
                title="Class별 예측 확률",
                range_y=[0, 1]
            )

            st.plotly_chart(
                fig_probability,
                use_container_width=True
            )

            st.caption(
                "예측 확률은 모델이 학습한 데이터의 패턴을 바탕으로 "
                "계산한 값이며, 실제 오류가 발생할 확률이나 "
                "현장 고장 위험도를 직접 의미하지는 않습니다."
            )

        st.divider()

        st.subheader("입력 데이터 확인")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        st.success(
            "새로운 센서 측정값의 Class 분류가 완료되었습니다."
        )

        st.warning(
            "이 결과는 학습에 사용된 데이터와 유사한 패턴을 "
            "가진 입력에 대한 실습용 분류 결과입니다. "
            "실제 산업용 로봇의 안전 판단이나 제어에 직접 사용해서는 안 됩니다."
        )


# =========================================================
# 15. 하단 안내
# =========================================================

st.divider()

st.caption("""
산업용 로봇 오류 유형 예측 실습실
데이터 출처: UCI Machine Learning Repository - Robot Execution Failures
프로젝트 분야: 기계공학 / 로봇·자동화 / 데이터 과학
""")
