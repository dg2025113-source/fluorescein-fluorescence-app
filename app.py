import io, json, uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

st.set_page_config(page_title="플루오레세인 형광 분석", page_icon="🧪", layout="wide")
st.title("🧪 플루오레세인 나트륨 형광 분석")
st.caption("사진 분석과 실험 기록을 같은 메뉴에서 진행합니다. 그래프는 직접 입력하거나 저장한 실험 데이터가 있을 때만 표시됩니다.")

if "records" not in st.session_state:
    st.session_state.records = []
if "conc_records" not in st.session_state:
    st.session_state.conc_records = []


def calc(img, l, r, t, b):
    a = np.asarray(img.convert("RGB")); h, w = a.shape[:2]
    x1, x2 = int(w*l/100), int(w*r/100)
    y1, y2 = int(h*t/100), int(h*b/100)
    roi = a[y1:y2, x1:x2, :]
    if roi.size == 0:
        raise ValueError("분석 영역(ROI) 범위를 확인하세요.")
    g = roi[:, :, 1]
    return {"mean_g": float(g.mean()), "max_g": int(g.max()), "sat": float((g == 255).mean()*100),
            "x1": x1, "x2": x2, "y1": y1, "y2": y2, "rw": x2-x1, "rh": y2-y1, "w": w, "h": h}


def preview(img, m):
    out = img.convert("RGB").copy()
    ImageDraw.Draw(out).rectangle((m["x1"], m["y1"], m["x2"]-1, m["y2"]-1), outline="red", width=max(2, out.width//300))
    return out


def jpg_bytes(img):
    b = io.BytesIO(); img.convert("RGB").save(b, format="JPEG", quality=92); return b.getvalue()


def plot_points(df, x, y, title):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    df = df.dropna(subset=[x, y]).sort_values(x)
    ax.plot(df[x], df[y], marker="o")
    ax.set(title=title, xlabel=x, ylabel=y); ax.grid(alpha=.25); fig.tight_layout()
    return fig

menu_items = ["🧪 농도 선정", "📈 pH 기준시료-형광", "🧫 완충용액-형광"]
page = st.sidebar.radio("실험 메뉴", menu_items)

# Shared image-analysis panel embedded inside each experiment menu.
def photo_panel(experiment_type, show_ph=False, show_conc=False, show_addition=False):
    st.subheader("사진 분석 및 기록")
    st.write("사진을 올리고 이 화면에서 분석·기록하면 됩니다. 다른 메뉴로 이동할 필요가 없습니다.")
    file = st.file_uploader("형광 사진 업로드", type=["png", "jpg", "jpeg", "webp"], key=f"upload_{experiment_type}")
    if not file:
        st.info("먼저 측정한 형광 사진을 업로드하세요.")
        return
    img = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    c1, c2 = st.columns(2)
    sample_name = c1.text_input("시료 이름", placeholder="예: pH 6.2 반복 1", key=f"name_{experiment_type}")
    memo = c2.text_input("메모", placeholder="측정 조건, 관찰 사항 등", key=f"memo_{experiment_type}")
    c3, c4 = st.columns(2)
    ph = c3.number_input("측정 pH", min_value=0.0, max_value=14.0, value=0.0, step=0.1, key=f"ph_{experiment_type}") if show_ph else None
    conc = c4.text_input("농도", value="0.001%" if experiment_type == "농도 선정" else "", key=f"conc_{experiment_type}") if show_conc else None
    if show_addition:
        c5, c6 = st.columns(2)
        additive = c5.selectbox("첨가물", ["첨가 없음", "HCl", "NaOH", "기타"], key=f"add_{experiment_type}")
        amount = c6.number_input("첨가량 (mL)", min_value=0.0, value=0.0, step=0.5, key=f"amount_{experiment_type}")
    else:
        additive, amount = None, None
    st.markdown("**분석 영역(ROI) 설정**")
    c1, c2 = st.columns(2)
    l = c1.slider("왼쪽 경계 (%)", 0, 99, 35, key=f"l_{experiment_type}")
    r = c2.slider("오른쪽 경계 (%)", 1, 100, 65, key=f"r_{experiment_type}")
    c3, c4 = st.columns(2)
    t = c3.slider("위쪽 경계 (%)", 0, 99, 55, key=f"t_{experiment_type}")
    b = c4.slider("아래쪽 경계 (%)", 1, 100, 75, key=f"b_{experiment_type}")
    if l >= r or t >= b:
        st.error("ROI 경계를 확인하세요: 왼쪽은 오른쪽보다 작고, 위쪽은 아래쪽보다 작아야 합니다.")
        return
    m = calc(img, l, r, t, b); marked = preview(img, m)
    c1, c2 = st.columns(2)
    c1.image(img, caption="원본 사진", use_container_width=True)
    c2.image(marked, caption="빨간 사각형 = 분석 영역(ROI)", use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 G 밝기", f"{m['mean_g']:.2f}/255")
    c2.metric("최대 G 밝기", f"{m['max_g']}/255")
    c3.metric("포화 픽셀 비율", f"{m['sat']:.2f}%")
    st.caption(f"ROI 크기: {m['rw']} × {m['rh']} px. 평균 G 밝기는 이미지의 녹색 채널 기준 상대값이며, 절대 형광 세기 단위는 아닙니다.")

    if experiment_type == "농도 선정":
        st.markdown("#### 농도 적합성 기준")
        st.caption("기준은 실험 전에 정하고, 측정 결과에 맞춰 사후 변경하지 않는 것이 좋습니다.")
        c1, c2 = st.columns(2)
        low = c1.number_input("밝기 하한", 0.0, 254.0, 30.0, 1.0, key="conc_low")
        high = c2.number_input("밝기 상한", 1.0, 255.0, 245.0, 1.0, key="conc_high")
        c3, c4 = st.columns(2)
        satmax = c3.number_input("허용 포화 비율 상한 (%)", 0.0, 100.0, 1.0, 0.1, key="conc_satmax")
        bg = c4.number_input("배경 평균 G 밝기 (모르면 0)", 0.0, 255.0, 0.0, 0.1, key="conc_bg")
        c5, c6 = st.columns(2)
        repeats = c5.number_input("반복 측정 횟수", 1, 20, 1, key="conc_repeats")
        sd = c6.number_input("반복 측정 표준편차 (모르면 0)", 0.0, 255.0, 0.0, 0.01, key="conc_sd")
        sdmax = st.number_input("허용 표준편차", 0.0, 100.0, 5.0, 0.1, key="conc_sdmax")
        checks = [("밝기 범위", "충족" if low <= m["mean_g"] <= high else "미충족"),
                  ("포화 비율", "충족" if m["sat"] <= satmax else "미충족"),
                  ("배경과 구분", ("충족" if m["mean_g"]-bg >= 10 else "미충족") if bg > 0 else "확인 필요"),
                  ("반복 안정성", ("충족" if sd <= sdmax else "미충족") if repeats >= 2 else "확인 필요")]
        for label, status in checks:
            st.write(("✅" if status == "충족" else "❌" if status == "미충족" else "⚠️") + f" **{label}:** {status}")
        st.caption("배경 구분의 10 밝기 차이는 임시 기준입니다. 실제 실험 조건에 맞춰 사전에 정하세요.")
    else:
        checks = []

    rec = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
           "experiment_type": experiment_type, "sample_name": sample_name, "memo": memo,
           "ph": ph if show_ph and ph > 0 else None, "concentration": conc if show_conc else None,
           "additive": additive, "addition_ml": amount,
           "mean_g": round(m["mean_g"], 4), "max_g": m["max_g"], "saturation_pct": round(m["sat"], 4),
           "roi_left_pct": l, "roi_right_pct": r, "roi_top_pct": t, "roi_bottom_pct": b,
           "roi_width": m["rw"], "roi_height": m["rh"], "image_width": m["w"], "image_height": m["h"]}
    st.download_button("현재 측정 결과 CSV 다운로드", pd.DataFrame([{k:v for k,v in rec.items()}]).to_csv(index=False).encode("utf-8-sig"),
                       f"{experiment_type.replace(' ', '_')}_result.csv", "text/csv", key=f"csv_{experiment_type}")
    st.download_button("분석 영역 표시 사진 다운로드", jpg_bytes(marked), f"{experiment_type.replace(' ', '_')}_roi.jpg", "image/jpeg", key=f"jpg_{experiment_type}")
    if st.button("💾 이 측정 결과 저장", type="primary", key=f"save_{experiment_type}"):
        rec["image_bytes"] = jpg_bytes(img)
        st.session_state.records.insert(0, rec)
        if experiment_type == "농도 선정":
            conc_row = {"시료 이름": sample_name, "농도": conc, "평균 G 밝기": round(m["mean_g"], 4), "포화 비율(%)": round(m["sat"], 4),
                        **{k:v for k,v in checks}, "반복 횟수": repeats, "표준편차": sd if repeats >= 2 else None}
            st.session_state.conc_records.insert(0, conc_row)
        st.success("현재 세션에 저장했어요. 다른 메뉴로 이동해도 이 세션에서는 기록을 확인할 수 있습니다. 중요한 결과는 파일로 다운로드해 보관하세요.")

if page == "🧪 농도 선정":
    st.header("농도 선정")
    st.write("사진 업로드, 밝기 분석, 포화도 확인, 농도 적합성 판정을 한 화면에서 진행합니다.")
    photo_panel("농도 선정", show_conc=True)

elif page == "📈 pH 기준시료-형광":
    st.header("pH 기준시료-형광")
    st.write("pH가 알려진 기준 시료의 사진을 분석해 pH와 형광 밝기의 관계를 기록합니다.")
    photo_panel("pH 기준시료-형광", show_ph=True, show_conc=True)
    rows = [r for r in st.session_state.records if r["experiment_type"] == "pH 기준시료-형광" and r.get("ph") is not None]
    if rows:
        st.subheader("저장된 pH 기준시료 데이터 그래프")
        df = pd.DataFrame(rows).rename(columns={"ph":"pH", "mean_g":"평균 G 밝기"})
        st.pyplot(plot_points(df, "pH", "평균 G 밝기", "pH와 형광 밝기의 관계 (저장된 측정값)"))
        st.download_button("pH 기준시료 데이터 CSV 다운로드", df[["sample_name", "pH", "concentration", "mean_g", "saturation_pct", "memo"]].to_csv(index=False).encode("utf-8-sig"), "ph_reference_data.csv", "text/csv")
    else:
        st.info("아직 저장된 pH 기준시료 측정값이 없습니다. 사진을 분석하고 ‘이 측정 결과 저장’을 누르면 실제 측정값으로 그래프가 생성됩니다.")

elif page == "🧫 완충용액-형광":
    st.header("완충용액-형광")
    st.write("완충용액에 산 또는 염기를 첨가한 뒤 사진을 분석하고, 첨가량·pH와 형광 밝기를 함께 기록합니다.")
    photo_panel("완충용액-형광", show_ph=True, show_conc=True, show_addition=True)
    rows = [r for r in st.session_state.records if r["experiment_type"] == "완충용액-형광"]
    if rows:
        st.subheader("저장된 완충용액 데이터 그래프")
        df = pd.DataFrame(rows).rename(columns={"ph":"pH", "mean_g":"평균 G 밝기", "addition_ml":"첨가량 (mL)"})
        valid_add = df.dropna(subset=["첨가량 (mL)", "평균 G 밝기"])
        if len(valid_add):
            for additive in [x for x in valid_add["additive"].dropna().unique()]:
                sub = valid_add[valid_add["additive"] == additive]
                if len(sub) >= 1:
                    st.write(f"**{additive}**")
                    st.pyplot(plot_points(sub, "첨가량 (mL)", "평균 G 밝기", f"첨가량과 형광 밝기 — {additive}"))
        valid_ph = df.dropna(subset=["pH", "평균 G 밝기"])
        if len(valid_ph):
            st.pyplot(plot_points(valid_ph, "pH", "평균 G 밝기", "pH와 형광 밝기 (저장된 측정값)"))
        st.download_button("완충용액 데이터 CSV 다운로드", df.drop(columns=["image_bytes"], errors="ignore").to_csv(index=False).encode("utf-8-sig"), "buffer_fluorescence_data.csv", "text/csv")
    else:
        st.info("아직 저장된 완충용액 측정값이 없습니다. 사진을 분석하고 ‘이 측정 결과 저장’을 누르면 실제 측정값으로 그래프가 생성됩니다.")


with st.expander("🗂️ 저장 기록 / 데이터 다운로드", expanded=False):
    st.warning("이 버전은 임시 저장 방식입니다. 앱이나 세션이 초기화되면 기록이 사라질 수 있으니 중요한 결과는 CSV와 사진으로 다운로드해 보관하세요.")
    if not st.session_state.records:
        st.info("아직 저장된 측정 결과가 없습니다. 위 메뉴에서 사진을 분석하고 저장하세요.")
    for i, rec in enumerate(st.session_state.records):
        with st.expander(f"{rec.get('experiment_type')} · {rec.get('sample_name') or '이름 없는 시료'} · G {rec.get('mean_g')}"):
            for k, label in [("ph", "pH"), ("concentration", "농도"), ("additive", "첨가물"), ("addition_ml", "첨가량 (mL)"), ("mean_g", "평균 G 밝기"), ("max_g", "최대 G 밝기"), ("saturation_pct", "포화 비율 (%)"), ("memo", "메모")]:
                st.write(f"**{label}:** {rec.get(k) if rec.get(k) not in (None, '') else '미입력'}")
            if rec.get("image_bytes"):
                st.image(rec["image_bytes"], use_container_width=True)
            json_data = {k:v for k,v in rec.items() if k != "image_bytes"}
            st.download_button("이 기록 JSON 다운로드", json.dumps(json_data, ensure_ascii=False, indent=2).encode("utf-8"), f"measurement_{i+1}.json", "application/json", key=f"record_json_{i}")
    if st.session_state.records:
        rows = [{k:v for k,v in r.items() if k != "image_bytes"} for r in st.session_state.records]
        st.download_button("전체 측정 기록 CSV 다운로드", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"), "all_measurements.csv", "text/csv")
    if st.session_state.conc_records:
        st.subheader("농도 선정 판정 기록")
        df = pd.DataFrame(st.session_state.conc_records)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("농도 선정 판정 CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"), "concentration_selection.csv", "text/csv")
