import streamlit as st
from pyswip import Prolog
from model import analyze_image
import os
import sys
import json

# Đảm bảo mã hóa UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

def safe_decode(text):
    if isinstance(text, bytes):
        try:
            return text.decode("utf-8")
        except UnicodeDecodeError:
            return text.decode("latin-1", errors="replace")
    return str(text)

def integrate_ml_prolog(image_path, medical_history={}):
    ml_result = analyze_image(image_path)
    symptoms = ml_result["symptoms"]
    prolog = Prolog()
    prolog.consult("medical_rules.pl")

    debug_info = []
    debug_info.append(f"Triệu chứng từ mô hình: {symptoms}")

    # Thêm các triệu chứng từ model, chuyển sang tiếng Việt không dấu
    symptom_map = {
        "cough": "ho",
        "fever": "sot",
        "chest_pain": "dau_nguc"
    }
    for sym in symptoms:
        sym_vn = symptom_map.get(sym, sym)  # Chuyển sang tên tiếng Việt
        prolog.assertz(f"trieu_chung({sym_vn})")
        debug_info.append(f"Đã thêm từ mô hình: trieu_chung({sym_vn})")

    # Thêm triệu chứng bổ sung
    if medical_history.get("fatigue", False):
        prolog.assertz("trieu_chung(met_moi)")
        debug_info.append("Đã thêm: trieu_chung(met_moi)")
    if medical_history.get("shortness_of_breath", False):
        prolog.assertz("trieu_chung(kho_tho)")
        debug_info.append("Đã thêm: trieu_chung(kho_tho)")

    # Query tất cả chẩn đoán
    results = list(prolog.query("chan_doan(D), dieu_tri(D, T)"))

    # Debug: Kiểm tra trieu_chung sau assert
    current_syms = list(prolog.query("trieu_chung(X)"))
    debug_info.append(f"Các triệu chứng hiện tại trong Prolog: {current_syms}")

    if results:
        diagnoses = []
        seen = set()
        for result in results:
            diagnosis = safe_decode(result['D'])
            treatment = safe_decode(result['T'])
            if diagnosis not in seen:
                diagnoses.append(f"{diagnosis}: {treatment}")
                seen.add(diagnosis)

        # Ưu tiên covid nếu có khó thở
        treatment = [
            d.split(": ")[1]
            for d in diagnoses
            if "covid" in d and medical_history.get("shortness_of_breath", False)
        ]
        if treatment:
            return {"chan_doan": ", ".join(diagnoses), "dieu_tri": treatment[0], "debug": debug_info}

        return {
            "chan_doan": ", ".join(diagnoses),
            "dieu_tri": diagnoses[0].split(": ")[1] if diagnoses else "Tu van bac si",
            "debug": debug_info
        }
    else:
        return {"chan_doan": "khong_xac_dinh", "dieu_tri": "Tu van bac si", "debug": debug_info}

# =======================
# Giao diện Streamlit
# =======================
st.set_page_config(page_title="He Thong Y Te Thong Minh", layout="wide")

# Chia cột
col1, col2 = st.columns([2, 1])
with col1:
    st.title("He Thong Y Te Thong Minh")
    uploaded_file = st.file_uploader("Chon anh X-quang", type=["jpg", "jpeg", "png"])

    # Hiển thị placeholder nếu chưa upload
    if not uploaded_file and os.path.exists("placeholder.jpg"):
        st.image("placeholder.jpg", caption="Chua upload anh", use_container_width=True)
    elif uploaded_file:
        image_path = "temp.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Anh X-quang da upload", use_container_width=True)
        if os.path.exists(image_path):
            os.remove(image_path)

with col2:
    st.write("**Trieu chung bo sung**")
    fatigue = st.checkbox("Co trieu chung met moi")
    shortness_of_breath = st.checkbox("Co trieu chung kho tho")

    if st.button("Chan doan"):
        if uploaded_file is not None:
            image_path = "temp.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            medical_history = {
                "fatigue": fatigue,
                "shortness_of_breath": shortness_of_breath,
            }
            st.write("Dang su dung model da train tu trained_model.pth...")
            with st.spinner("Dang phan tich..."):
                result = integrate_ml_prolog(image_path, medical_history)

            # Kết quả chính
            st.success("Ket qua chan doan:")
            st.markdown("**Chan doan:** " + result['chan_doan'], unsafe_allow_html=True)
            st.markdown("**Dieu tri:** " + result['dieu_tri'], unsafe_allow_html=True)

            # Hiển thị quá trình suy luận
            with st.expander("Xem qua trinh suy luan chi tiet"):
                for line in result.get("debug", []):
                    st.write(line)

            # Lưu lịch sử
            with open("history.json", "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": uploaded_file.name,
                        "result": result,
                        "timestamp": st.session_state.get("timestamp", "N/A"),
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

            if os.path.exists(image_path):
                os.remove(image_path)
        else:
            st.error("Vui long upload anh X-quang truoc!")