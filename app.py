import io
import json
import uuid
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="플루오레세인 형광 분석", page_icon="🧪", layout="wide")
st.title("🧪 플루오레세인 나트륨 형광 분석")
st.caption("사진 분석과 실험 기록을 같은 메뉴에서 진행합니다. 그래프는 직접 저장한 실험 데이터가 있을 때만 표시됩니다.")

if "records" not in st.session_state:
    st.session_state.records = []
if "conc_records" not in st.session_state:
    st.session_state.conc_records = []


def calc(img, left_pct, right_pct, top_pct, bottom_pct):
    """선택한 ROI에서 G 채널 평균, 최대값, G=255 포화 픽셀 비율을 계산한다."""
    rgb = np.asarray(img.convert("RGB"))
    height, width = rgb.shape[:2]
    x1, x2 = int(width * left_pct / 100), int(width * right_pct / 100)
    y1, y2 = int(height * top_pct / 100), int(height * bottom_pct / 100)
    roi = rgb[y1:y2, x1:x2, :]
    if roi.size == 0:
        raise ValueError("분석 영역(ROI) 범위를 확인하세요.")
    g = roi[:, :, 1]
    return {
        "mean_g": float(g.mean()),
        "max_g": int(g.max()),
        "saturation_pct": float((g == 255).mean() * 100),
        "x1": x1, "x2": x2, "y1": y1, "y2": y2,
        "roi_width": x2 - x1, "roi_height": y2 - y1,
        "image_width": width, "image_height": height,
    }


def preview(img, metrics):
    out = img.convert("RGB").copy()
    ImageDraw.Draw(out).rectangle(
        (metrics["x1"], metrics["y1"], metrics["x2"] - 1, metrics["y2"] - 1),
        outline="red", width=max(2, out.width // 300)
    )
    return out


def jpg_bytes(img):
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def plot_points(df, x, y, title):
    plot_df = df.dropna(subset=[x, y]).copy().sort_values(x)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(plot_df[x], plot_df[y], marker="o")
    ax.set(title=title, xlabel=x, ylabel=y)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


menu_items = ["🧪 농도 선정", "📈 pH 기준시료-형광", "🧫 완충용액-형광"]
page = st.sidebar.radio("실험 메뉴", menu_items)


def photo_panel(experiment_type, show_ph=False, show_conc=False, show_addition=False):
    is_concentration = experiment_type == "농도 선정"
    st.subheader("사진 분석 및 기록")
    st.write("사진을 업로드하고 이 화면에서 분석·기록하면 됩니다. 다른 메뉴로 이동할 필요가 없습니다.")
    uploaded = st.file_uploader(
        "형광 사진 업로드", type=["png", "jpg", "jpeg", "webp"],
        key=f"upload_{experiment_type}"
    )
    if uploaded is None:
        st.info("먼저 측정한 형광 사진을 업로드하세요.")
        return

    try:
        img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    except Exception:
        st.error("이미지를 읽을 수 없습니다. PNG 또는 JPG 사진을 다시 업로드하세요.")
        return

    col1, col2 = st.columns(2)
    sample_name = col1.text_input("시료 이름", placeholder="예: pH 6.2 기준시료", key=f"name_{experiment_type}")
    memo = col2.text_input("메모", placeholder="측정 조건, 관찰 사항 등", key=f"memo_{experiment_type}")

    col3, col4 = st.columns(2)
    ph = col3.number_input("측정 pH", min_value=0.0, max_value=14.0, value=0.0, step=0.1, key=f"ph_{experiment_type}") if show_ph else None
    conc = col4.text_input("농도", value="0.001%" if is_concentration else "", key=f"conc_{experiment_type}") if show_conc else None

    if show_addition:
        col5, col6 = st.columns(2)
        additive = col5.selectbox("첨가물", ["첨가 없음", "HCl", "NaOH", "기타"], key=f"add_{experiment_type}")
        amount = col6.number_input("첨가량 (mL)", min_value=0.0, value=0.0, step=0.5, key=f"amount_{experiment_type}")
    else:
        additive, amount = None, None

    st.markdown("**분석 영역(ROI) 설정**")
    col1, col2 = st.columns(2)
    left = col1.slider("왼쪽 경계 (%)", 0, 99, 35, key=f"l_{experiment_type}")
    right = col2.slider("오른쪽 경계 (%)", 1, 100, 65, key=f"r_{experiment_type}")
    col3, col4 = st.columns(2)
    top = col3.slider("위쪽 경계 (%)", 0, 99, 55, key=f"t_{experiment_type}")
    bottom = col4.slider("아래쪽 경계 (%)", 1, 100, 75, key=f"b_{experiment_type}")

    if left >= right or top >= bottom:
        st.error("ROI 경계를 확인하세요: 왼쪽은 오른쪽보다 작고, 위쪽은 아래쪽보다 작아야 합니다.")
        return

    try:
        metrics = calc(img, left, right, top, bottom)
    except ValueError as error:
        st.error(str(error))
        return

    marked = preview(img, metrics)
    col1, col2 = st.columns(2)
    col1.image(img, caption="원본 사진", use_container_width=True)
    col2.image(marked, caption="빨간 사각형 = 분석 영역(ROI)", use_container_width=True)

    if is_concentration:
        st.metric("G=255 포화 픽셀 비율", f"{metrics['saturation_pct']:.2f}%")
        st.caption(
            f"ROI 크기: {metrics['roi_width']} × {metrics['roi_height']} px. "
            "포화 픽셀은 분석 영역에서 G 채널 값이 정확히 255인 픽셀입니다."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("평균 G 밝기", f"{metrics['mean_g']:.2f}/255")
        col2.metric("실제 최대 G 밝기", f"{metrics['max_g']}/255")
        col3.metric("G=255 포화 픽셀 비율", f"{metrics['saturation_pct']:.2f}%")
        st.caption(
            f"ROI 크기: {metrics['roi_width']} × {metrics['roi_height']} px. "
            "평균 G 밝기는 이미지의 녹색 채널 기준 상대값이며, 절대 형광 세기 단위가 아닙니다."
        )

    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": experiment_type,
        "sample_name": sample_name,
        "memo": memo,
        "ph": ph if show_ph and ph > 0 else None,
        "concentration": conc if show_conc else None,
        "additive": additive,
        "addition_ml": amount,
        "saturation_pct": round(metrics["saturation_pct"], 4),
        "roi_left_pct": left, "roi_right_pct": right,
        "roi_top_pct": top, "roi_bottom_pct": bottom,
        "roi_width": metrics["roi_width"], "roi_height": metrics["roi_height"],
        "image_width": metrics["image_width"], "image_height": metrics["image_height"],
    }
    # 농도 선정 메뉴에서는 평균/최대 G 밝기를 기록하지 않는다.
    if not is_concentration:
        record["mean_g"] = round(metrics["mean_g"], 4)
        record["max_g"] = metrics["max_g"]

    current_csv = pd.DataFrame([record]).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 측정 결과 CSV 다운로드", current_csv,
        f"{experiment_type.replace(' ', '_')}_result.csv", "text/csv",
        key=f"csv_{experiment_type}"
    )
    st.download_button(
        "분석 영역 표시 사진 다운로드", jpg_bytes(marked),
        f"{experiment_type.replace(' ', '_')}_roi.jpg", "image/jpeg",
        key=f"jpg_{experiment_type}"
    )

    if st.button("💾 이 측정 결과 저장", type="primary", key=f"save_{experiment_type}"):
        saved_record = record.copy()
        saved_record["image_bytes"] = jpg_bytes(img)
        st.session_state.records.insert(0, saved_record)
        if is_concentration:
            st.session_state.conc_records.insert(0, {
                "시료 이름": sample_name,
                "농도": conc,
                "G=255 포화 픽셀 비율(%)": round(metrics["saturation_pct"], 4),
            })
        st.success("현재 세션에 저장했어요. 중요한 결과는 CSV와 사진으로 다운로드해 보관하세요.")


# 농도 선정
if page == "🧪 농도 선정":
    st.header("농도 선정")
    st.write("분석 영역(ROI)에서 G 채널 값이 255인 포화 픽셀의 비율만 확인합니다.")
    photo_panel("농도 선정", show_conc=True)


# pH 기준시료-형광
elif page == "📈 pH 기준시료-형광":
    st.header("pH 기준시료-형광")
    st.write("pH가 알려진 기준 시료의 사진을 분석해 pH와 형광 밝기의 관계를 기록합니다.")
    photo_panel("pH 기준시료-형광", show_ph=True, show_conc=True)

    rows = [
        r for r in st.session_state.records
        if r["experiment_type"] == "pH 기준시료-형광" and r.get("ph") is not None
    ]
    if rows:
        st.subheader("저장된 pH 기준시료 데이터 그래프")
        df = pd.DataFrame(rows).rename(columns={"ph": "pH", "mean_g": "평균 G 밝기"})
        st.pyplot(plot_points(df, "pH", "평균 G 밝기", "pH와 형광 밝기의 관계 (저장된 측정값)"))

        # rename한 열 이름과 CSV 선택 열 이름을 일치시킨다.
        csv_columns = [
            "sample_name", "pH", "concentration",
            "평균 G 밝기", "max_g", "saturation_pct", "memo"
        ]
        for column in csv_columns:
            if column not in df.columns:
                df[column] = None
        ph_csv = df[csv_columns].to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "pH 기준시료 데이터 CSV 다운로드",
            ph_csv, "ph_reference_data.csv", "text/csv"
        )
    else:
        st.info(
            "아직 저장된 pH 기준시료 측정값이 없습니다. "
            "사진을 분석하고 ‘이 측정 결과 저장’을 누르면 실제 측정값으로 그래프가 생성됩니다."
        )


# 완충용액-형광
elif page == "🧫 완충용액-형광":
    st.header("완충용액-형광")
    st.write("완충용액에 산 또는 염기를 첨가한 뒤 사진을 분석하고, 첨가량·pH와 형광 밝기를 함께 기록합니다.")
    photo_panel("완충용액-형광", show_ph=True, show_conc=True, show_addition=True)

    rows = [r for r in st.session_state.records if r["experiment_type"] == "완충용액-형광"]
    if rows:
        st.subheader("저장된 완충용액 데이터 그래프")
        df = pd.DataFrame(rows).rename(columns={
            "ph": "pH",
            "mean_g": "평균 G 밝기",
            "addition_ml": "첨가량 (mL)",
        })

        valid_add = df.dropna(subset=["첨가량 (mL)", "평균 G 밝기"])
        if len(valid_add):
            for additive_name in valid_add["additive"].dropna().unique():
                subset = valid_add[valid_add["additive"] == additive_name]
                st.write(f"**{additive_name}**")
                st.pyplot(plot_points(
                    subset, "첨가량 (mL)", "평균 G 밝기",
                    f"첨가량과 형광 밝기 — {additive_name}"
                ))

        valid_ph = df.dropna(subset=["pH", "평균 G 밝기"])
        if len(valid_ph):
            st.pyplot(plot_points(
                valid_ph, "pH", "평균 G 밝기",
                "pH와 형광 밝기 (저장된 측정값)"
            ))

        buffer_csv = df.drop(columns=["image_bytes"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "완충용액 데이터 CSV 다운로드",
            buffer_csv, "buffer_fluorescence_data.csv", "text/csv"
        )
    else:
        st.info(
            "아직 저장된 완충용액 측정값이 없습니다. "
            "사진을 분석하고 ‘이 측정 결과 저장’을 누르면 실제 측정값으로 그래프가 생성됩니다."
        )


# 저장 기록 / 데이터 다운로드
with st.expander("🗂️ 저장 기록 / 데이터 다운로드", expanded=False):
    st.warning(
        "이 버전은 임시 저장 방식입니다. 앱이나 세션이 초기화되면 기록이 사라질 수 있으니 "
        "중요한 결과는 CSV와 사진으로 다운로드해 보관하세요."
    )

    if not st.session_state.records:
        st.info("아직 저장된 측정 결과가 없습니다. 위 메뉴에서 사진을 분석하고 저장하세요.")

    for index, record in enumerate(st.session_state.records):
        is_conc = record.get("experiment_type") == "농도 선정"
        sample = record.get("sample_name") or "이름 없는 시료"
        if is_conc:
            title = f"{record.get('experiment_type')} · {sample} · 포화 {record.get('saturation_pct')}%"
        else:
            title = f"{record.get('experiment_type')} · {sample} · G {record.get('mean_g')}"

        with st.expander(title):
            if is_conc:
                fields = [
                    ("concentration", "농도"),
                    ("saturation_pct", "G=255 포화 픽셀 비율 (%)"),
                    ("memo", "메모"),
                ]
            else:
                fields = [
                    ("ph", "pH"),
                    ("concentration", "농도"),
                    ("additive", "첨가물"),
                    ("addition_ml", "첨가량 (mL)"),
                    ("mean_g", "평균 G 밝기"),
                    ("max_g", "실제 최대 G 밝기"),
                    ("saturation_pct", "G=255 포화 픽셀 비율 (%)"),
                    ("memo", "메모"),
                ]

            for key, label in fields:
                value = record.get(key)
                st.write(f"**{label}:** {value if value not in (None, '') else '미입력'}")

            if record.get("image_bytes"):
                st.image(record["image_bytes"], use_container_width=True)

            json_data = {k: v for k, v in record.items() if k != "image_bytes"}
            st.download_button(
                "이 기록 JSON 다운로드",
                json.dumps(json_data, ensure_ascii=False, indent=2).encode("utf-8"),
                f"measurement_{index + 1}.json",
                "application/json",
                key=f"record_json_{index}"
            )

    if st.session_state.records:
        all_rows = [
            {k: v for k, v in record.items() if k != "image_bytes"}
            for record in st.session_state.records
        ]
        all_csv = pd.DataFrame(all_rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "전체 측정 기록 CSV 다운로드",
            all_csv, "all_measurements.csv", "text/csv"
        )

    if st.session_state.conc_records:
        st.subheader("농도 선정 포화 픽셀 기록")
        conc_df = pd.DataFrame(st.session_state.conc_records)
        st.dataframe(conc_df, use_container_width=True, hide_index=True)
        conc_csv = conc_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "농도 선정 포화 픽셀 CSV 다운로드",
            conc_csv, "concentration_selection.csv", "text/csv"
        )
