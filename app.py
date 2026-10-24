import io, json, uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

try:
    from supabase import create_client
except Exception:
    create_client = None

st.set_page_config(page_title="플루오레세인 형광 분석", page_icon="🧪", layout="wide")
st.title("🧪 플루오레세인 나트륨 형광 분석")
st.caption("평균 G 채널 밝기(0–255)는 상대적 디지털 밝기값이며, 물리적 형광 세기 단위가 아닙니다.")

if "local_records" not in st.session_state: st.session_state.local_records = []
if "conc_records" not in st.session_state: st.session_state.conc_records = []

def sb():
    try:
        if create_client and st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_KEY"):
            return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]), st.secrets.get("SUPABASE_BUCKET","fluorescence-images")
    except Exception: pass
    return None

def jpg_bytes(img):
    b=io.BytesIO(); img.convert("RGB").save(b,format="JPEG",quality=92); return b.getvalue()

def calc(img,l,r,t,b):
    a=np.asarray(img.convert("RGB")); h,w=a.shape[:2]
    x1,x2=int(w*l/100),int(w*r/100); y1,y2=int(h*t/100),int(h*b/100)
    roi=a[y1:y2,x1:x2,:]
    if roi.size==0: raise ValueError("ROI 범위를 확인하세요.")
    g=roi[:,:,1]
    return {"mean_g":float(g.mean()),"max_g":int(g.max()),"sat":float((g==255).mean()*100),
            "x1":x1,"x2":x2,"y1":y1,"y2":y2,"rw":x2-x1,"rh":y2-y1,"w":w,"h":h}

def preview(img,m):
    out=img.convert("RGB").copy(); ImageDraw.Draw(out).rectangle((m["x1"],m["y1"],m["x2"]-1,m["y2"]-1),outline="red",width=max(2,out.width//300)); return out

def save_record(rec, imgbytes):
    cloud=sb()
    if cloud:
        client,bucket=cloud
        path=f"{rec['id']}.jpg"
        client.storage.from_(bucket).upload(path,imgbytes,file_options={"content-type":"image/jpeg","upsert":"true"})
        payload={k:v for k,v in rec.items() if k!="image_bytes"}; payload["image_path"]=path
        client.table("fluorescence_records").insert(payload).execute()
        return "cloud"
    st.session_state.local_records.insert(0,{**rec,"image_bytes":imgbytes})
    return "session"

def lineplot(df,x,y,title,series=None):
    fig,ax=plt.subplots(figsize=(8,4.3))
    if series and series in df.columns:
        for name,g in df.groupby(series):
            g=g.sort_values(x); ax.plot(g[x],g[y],marker="o",label=str(name))
        ax.legend()
    else:
        g=df.sort_values(x); ax.plot(g[x],g[y],marker="o")
    ax.set(title=title,xlabel=x,ylabel=y); ax.grid(alpha=.25); fig.tight_layout(); return fig

page=st.sidebar.radio("메뉴",["📷 사진 분석","🧪 농도 선정","📊 실험 그래프","🖼️ 저장 갤러리","🗃️ 데이터 다운로드"])

if page=="📷 사진 분석":
    st.header("사진 분석")
    file=st.file_uploader("사진 업로드",type=["png","jpg","jpeg","webp"])
    if file:
        img=Image.open(io.BytesIO(file.getvalue())).convert("RGB")
        c1,c2=st.columns(2)
        name=c1.text_input("시료 이름",placeholder="예: pH 7.3")
        kind=c2.selectbox("실험 종류",["일반 pH 측정","농도 선정","완충용액","기타"])
        c3,c4=st.columns(2)
        ph=c3.number_input("pH (모르면 0)",0.0,14.0,0.0,0.1)
        conc=c4.text_input("농도",placeholder="예: 0.001%")
        memo=st.text_input("메모")
        st.subheader("ROI 설정")
        l=st.slider("왼쪽 경계 (%)",0,99,35); r=st.slider("오른쪽 경계 (%)",1,100,65)
        t=st.slider("위쪽 경계 (%)",0,99,55); b=st.slider("아래쪽 경계 (%)",1,100,75)
        if l>=r or t>=b: st.error("ROI 경계를 확인하세요."); st.stop()
        m=calc(img,l,r,t,b); marked=preview(img,m)
        c1,c2=st.columns(2); c1.image(img,caption="원본",use_container_width=True); c2.image(marked,caption="빨간 사각형 = ROI",use_container_width=True)
        a,bx,c=st.columns(3); a.metric("평균 G 밝기",f"{m['mean_g']:.2f}/255"); bx.metric("최대 G 밝기",f"{m['max_g']}/255"); c.metric("포화 픽셀 비율",f"{m['sat']:.2f}%")
        st.caption(f"ROI: {m['rw']}×{m['rh']} px · 원본: {m['w']}×{m['h']} px. ROI는 이미지 너비·높이에 대한 백분율로 저장됩니다.")
        rec={"id":str(uuid.uuid4()),"created_at":datetime.now(timezone.utc).isoformat(),"sample_name":name,"experiment_type":kind,"ph":ph or None,"concentration":conc,"memo":memo,"mean_g":round(m["mean_g"],4),"max_g":m["max_g"],"saturation_pct":round(m["sat"],4),"roi_left_pct":l,"roi_right_pct":r,"roi_top_pct":t,"roi_bottom_pct":b,"roi_width":m["rw"],"roi_height":m["rh"],"image_width":m["w"],"image_height":m["h"]}
        st.download_button("현재 결과 CSV 다운로드",pd.DataFrame([rec]).to_csv(index=False).encode("utf-8-sig"),"fluorescence_result.csv","text/csv")
        st.download_button("ROI 표시 사진 다운로드",jpg_bytes(marked),"roi_preview.jpg","image/jpeg")
        if st.button("💾 결과 저장",type="primary"):
            try:
                mode=save_record(rec,jpg_bytes(img))
                st.success("클라우드에 저장했습니다." if mode=="cloud" else "현재 세션에 저장했습니다. 영구 저장은 Supabase 설정이 필요합니다.")
            except Exception as e: st.error(f"저장 실패: {e}")
        st.info("포화 비율은 ROI에서 G 채널 값이 255인 픽셀의 비율입니다. 카메라 포화를 완벽하게 진단하는 지표는 아닙니다.")

elif page=="🧪 농도 선정":
    st.header("플루오레세인 나트륨 농도 선정")
    st.write("그래프 없이 조건별 판정을 확인합니다. 기준은 실험 전에 정하고, 결과를 본 뒤 유리하게 바꾸지 마세요.")
    with st.form("concform"):
        c1,c2=st.columns(2); conc=c1.text_input("농도", "0.001%"); mean=c2.number_input("평균 G 밝기",0.0,255.0,192.58,0.01)
        c3,c4=st.columns(2); sat=c3.number_input("포화 픽셀 비율 (%)",0.0,100.0,0.0,0.01); bg=c4.number_input("배경 평균 G (모르면 0)",0.0,255.0,0.0,0.1)
        low=st.number_input("밝기 하한",0.0,254.0,30.0,1.0); high=st.number_input("밝기 상한",1.0,255.0,245.0,1.0)
        satmax=st.number_input("허용 포화 비율 상한 (%)",0.0,100.0,1.0,0.1)
        n=st.number_input("반복 측정 횟수",1,20,1); sd=st.number_input("반복 측정 표준편차 (모르면 0)",0.0,255.0,0.0,0.01); sdmax=st.number_input("허용 표준편차",0.0,100.0,5.0,0.1)
        go=st.form_submit_button("조건 판정 및 기록")
    if go:
        checks=[("밝기 범위","충족" if low<=mean<=high else "미충족"),("포화 비율","충족" if sat<=satmax else "미충족"),("배경과 구분",("충족" if mean-bg>=10 else "미충족") if bg>0 else "확인 필요"),("반복 안정성",("충족" if sd<=sdmax else "미충족") if n>=2 else "확인 필요")]
        row={"농도":conc,"평균 G 밝기":mean,"포화 비율(%)":sat,**{k:v for k,v in checks},"반복 횟수":n,"표준편차":sd if n>=2 else None}
        st.session_state.conc_records.insert(0,row)
        for label,status in checks: st.write(("✅" if status=="충족" else "❌" if status=="미충족" else "⚠️")+f" **{label}:** {status}")
        st.info("모든 항목이 충족되어야 최종 선정 근거가 충분합니다. 배경 구분은 ‘시료가 배경보다 G 밝기 10 이상 높음’이라는 임시 기준입니다.")
    if st.session_state.conc_records:
        df=pd.DataFrame(st.session_state.conc_records); st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("농도 기록 CSV 다운로드",df.to_csv(index=False).encode("utf-8-sig"),"concentration_selection.csv","text/csv")

elif page=="📊 실험 그래프":
    st.header("실험 그래프")
    tab1,tab2,tab3=st.tabs(["pH–형광 밝기","완충용액","CSV 미리보기"])
    with tab1:
        df=pd.DataFrame({"pH":[4.1,5.4,6.2,7.3,8.0,8.9],"평균 G 밝기":[138.92,165.58,192.10,211.67,221.13,232.79]})
        df=st.data_editor(df,num_rows="dynamic",use_container_width=True,key="phdata")
        if len(df.dropna()): st.pyplot(lineplot(df.dropna(),"pH","평균 G 밝기","pH와 형광 밝기의 관계"))
        st.caption("pH 7.1의 측정값은 기존 기준곡선과 맞지 않아 기본 그래프에서 제외했습니다. 원자료는 별도로 보관하세요.")
        st.download_button("pH 데이터 CSV",df.to_csv(index=False).encode("utf-8-sig"),"ph_fluorescence.csv","text/csv")
    with tab2:
        df=pd.DataFrame({"첨가물":["HCl"]*7+["NaOH"]*7,"첨가량(mL)":[0,1,2,3,4,5,6]*2,"pH":[6.0,5.9,5.8,5.8,5.7,5.6,5.4,6.1,6.5,6.8,7.0,7.3,7.7,10.0],"평균 G 밝기":[157.04,183.80,174.77,172.19,176.99,166.53,127.90,127.27,136.46,140.91,131.13,146.30,159.93,141.41]})
        df=st.data_editor(df,num_rows="dynamic",use_container_width=True,key="bufdata")
        df=df.dropna()
        st.pyplot(lineplot(df,"첨가량(mL)","평균 G 밝기","첨가량과 형광 밝기","첨가물"))
        st.pyplot(lineplot(df,"pH","평균 G 밝기","pH와 형광 밝기","첨가물"))
        st.download_button("완충용액 CSV",df.to_csv(index=False).encode("utf-8-sig"),"buffer_fluorescence.csv","text/csv")
    with tab3:
        f=st.file_uploader("CSV 업로드",type=["csv"],key="csvview")
        if f:
            try: st.dataframe(pd.read_csv(f),use_container_width=True)
            except Exception as e: st.error(str(e))

elif page=="🖼️ 저장 갤러리":
    st.header("저장 갤러리")
    cloud=sb()
    if cloud:
        client,bucket=cloud
        try:
            records=client.table("fluorescence_records").select("*").order("created_at",desc=True).limit(100).execute().data or []
            mode="cloud"
        except Exception as e:
            st.error(f"클라우드 기록을 읽지 못했습니다: {e}"); records=[]; mode="error"
    else:
        records=st.session_state.local_records; mode="session"
        st.warning("현재 세션 저장 모드입니다. 앱 재시작 후 유지하려면 Supabase를 설정하세요.")
    if not records: st.info("저장된 결과가 없습니다. 사진 분석에서 결과 저장을 누르세요.")
    for i,rec in enumerate(records):
        with st.expander(f"{rec.get('sample_name') or rec.get('concentration') or '이름 없는 시료'} · G {rec.get('mean_g','—')} · {str(rec.get('created_at',''))[:19]}"):
            for k,label in [("experiment_type","실험 종류"),("ph","pH"),("concentration","농도"),("mean_g","평균 G 밝기"),("max_g","최대 G 밝기"),("saturation_pct","포화 비율"),("memo","메모")]:
                st.write(f"**{label}:** {rec.get(k) or '미입력'}")
            st.write(f"**ROI:** 좌 {rec.get('roi_left_pct')}%, 우 {rec.get('roi_right_pct')}%, 위 {rec.get('roi_top_pct')}%, 아래 {rec.get('roi_bottom_pct')}%")
            if mode=="session" and rec.get("image_bytes"): st.image(rec["image_bytes"],use_container_width=True)
            elif mode=="cloud" and rec.get("image_path"):
                try:
                    url=client.storage.from_(bucket).create_signed_url(rec["image_path"],3600)
                    st.image(url.get("signedURL") or url.get("signedUrl"),use_container_width=True)
                except Exception: st.caption("사진 미리보기를 불러오지 못했습니다.")
            st.download_button("기록 JSON 다운로드",json.dumps({k:v for k,v in rec.items() if k!="image_bytes"},ensure_ascii=False,indent=2).encode("utf-8"),f"record_{i+1}.json","application/json",key=f"json{i}")

else:
    st.header("데이터 다운로드")
    if st.session_state.local_records:
        rows=[{k:v for k,v in r.items() if k!="image_bytes"} for r in st.session_state.local_records]
        df=pd.DataFrame(rows); st.dataframe(df,use_container_width=True)
        st.download_button("분석 기록 CSV 다운로드",df.to_csv(index=False).encode("utf-8-sig"),"analysis_history.csv","text/csv")
    else: st.info("현재 세션에 저장된 분석 기록이 없습니다.")
    if st.session_state.conc_records:
        df=pd.DataFrame(st.session_state.conc_records); st.dataframe(df,use_container_width=True)
        st.download_button("농도 선정 CSV 다운로드",df.to_csv(index=False).encode("utf-8-sig"),"concentration_records.csv","text/csv")
    st.markdown("**영구 저장:** Streamlit Cloud는 임시 세션 저장을 보장하지 않습니다. Supabase를 연결하면 저장 갤러리가 재시작 후에도 유지됩니다.")
