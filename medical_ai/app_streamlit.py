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
    diagnosis = ml_result["diagnosis"]
    symptoms = ml_result["symptoms"]
    debug_info = [f"ML result: diagnosis={diagnosis}, symptoms={symptoms}"]

    # Nếu là NORMAL và không có triệu chứng bổ sung, trả về ngay
    if diagnosis == "NORMAL" and not medical_history.get("fatigue", False) and not medical_history.get("shortness_of_breath", False):
        debug_info.append("No symptoms from model or medical history, returning normal")
        return {
            "chan_doan": "normal",
            "dieu_tri": "Phổi bình thường, không cần điều trị",
            "debug": debug_info
        }

    prolog = Prolog()
    try:
        prolog.consult("medical_rules.pl")
    except Exception as e:
        debug_info.append(f"Lỗi khi load medical_rules.pl: {str(e)}")
        return {"chan_doan": "unknown", "dieu_tri": f"Lỗi Prolog: {str(e)}", "debug": debug_info}

    # Thêm triệu chứng từ mô hình
    for sym in symptoms:
        prolog.assertz(f"current_symptom({sym})")
        debug_info.append(f"Đã thêm từ mô hình: current_symptom({sym})")

    # Thêm triệu chứng bổ sung
    if medical_history.get("fatigue", False):
        prolog.assertz("current_symptom(fatigue)")
        debug_info.append("Đã thêm: current_symptom(fatigue)")
    if medical_history.get("shortness_of_breath", False):
        prolog.assertz("current_symptom(shortness_of_breath)")
        debug_info.append("Đã thêm: current_symptom(shortness_of_breath)")

    # Thêm normal_flag nếu diagnosis là NORMAL
    if diagnosis == "NORMAL":
        prolog.assertz("current_symptom(normal_flag)")
        debug_info.append("Đã thêm: current_symptom(normal_flag)")

    # Query chẩn đoán và điều trị
    try:
        results = list(prolog.query("diagnosis(D), treatment(D, T)"))
    except Exception as e:
        debug_info.append(f"Lỗi query Prolog: {str(e)}")
        return {"chan_doan": "unknown", "dieu_tri": "Tư vấn bác sĩ", "debug": debug_info}

    current_syms = list(prolog.query("current_symptom(X)"))
    debug_info.append(f"Các triệu chứng hiện tại trong Prolog: {current_syms}")

    if results:
        diagnoses = []
        seen = set()
        for result in results:
            diag = safe_decode(result['D'])
            treat = safe_decode(result['T'])
            if diag not in seen:
                diagnoses.append(f"{diag}: {treat}")
                seen.add(diag)

        # Ưu tiên covid nếu có shortness_of_breath
        treatment = [
            d.split(": ")[1]
            for d in diagnoses
            if "covid" in d and medical_history.get("shortness_of_breath", False)
        ]
        return {
            "chan_doan": ", ".join(diagnoses),
            "dieu_tri": treatment[0] if treatment else diagnoses[0].split(": ")[1],
            "debug": debug_info
        }
    return {"chan_doan": "unknown", "dieu_tri": "Tư vấn bác sĩ", "debug": debug_info}

# Giao diện Streamlit
st.set_page_config(page_title="Hệ Thống Y Tế Thông Minh", layout="wide")

col1, col2 = st.columns([2, 1])
with col1:
    st.title("Hệ Thống Y Tế Thông Minh")
    uploaded_file = st.file_uploader("Chọn ảnh X-quang", type=["jpg", "jpeg", "png"])

    if not uploaded_file and os.path.exists("placeholder.jpg"):
        st.image("placeholder.jpg", caption="Chưa upload ảnh", use_container_width=True)
    elif uploaded_file:
        image_path = "temp.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Ảnh X-quang đã upload", use_container_width=True)
        if os.path.exists(image_path):
            os.remove(image_path)

with col2:
    st.write("**Triệu chứng bổ sung**")
    fatigue = st.checkbox("Có triệu chứng mệt mỏi")
    shortness_of_breath = st.checkbox("Có triệu chứng khó thở")

    if st.button("Chẩn đoán"):
        if uploaded_file is not None:
            image_path = "temp.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            medical_history = {
                "fatigue": fatigue,
                "shortness_of_breath": shortness_of_breath,
            }
            st.write("Đang sử dụng model đã train từ trained_model.pth...")
            with st.spinner("Đang phân tích..."):
                result = integrate_ml_prolog(image_path, medical_history)

            st.success("Kết quả chẩn đoán:")
            st.markdown("**Chẩn đoán:** " + result['chan_doan'], unsafe_allow_html=True)
            st.markdown("**Điều trị:** " + result['dieu_tri'], unsafe_allow_html=True)

            with st.expander("Xem quá trình suy luận chi tiết"):
                for line in result.get("debug", []):
                    st.write(line)

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
            st.error("Vui lòng upload ảnh X-quang trước!")