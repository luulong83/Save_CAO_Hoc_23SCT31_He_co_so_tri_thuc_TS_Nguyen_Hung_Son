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
    debug_info.append(f"Symptoms from model: {symptoms}")

    # Thêm các triệu chứng từ model
    for sym in symptoms:
        prolog.assertz(f"current_symptom({sym})")
        debug_info.append(f"Asserted from model: current_symptom({sym})")

    # Thêm triệu chứng bổ sung
    if medical_history.get("fatigue", False):
        prolog.assertz("current_symptom(fatigue)")
        debug_info.append("Asserted: current_symptom(fatigue)")
    if medical_history.get("shortness_of_breath", False):
        prolog.assertz("current_symptom(shortness_of_breath)")
        debug_info.append("Asserted: current_symptom(shortness_of_breath)")

    # Query tất cả chẩn đoán
    results = list(prolog.query("diagnosis(D), treatment(D, T)"))

    # Debug: Kiểm tra current_symptom sau assert
    current_syms = list(prolog.query("current_symptom(X)"))
    debug_info.append(f"Current symptoms in Prolog: {current_syms}")

    if results:
        diagnoses = []
        seen = set()
        for result in results:
            diagnosis = safe_decode(result['D'])
            treatment = safe_decode(result['T'])
            if diagnosis not in seen:
                diagnoses.append(f"{diagnosis}: {treatment}")
                seen.add(diagnosis)

        # Ưu tiên covid nếu shortness_of_breath
        treatment = [
            d.split(": ")[1]
            for d in diagnoses
            if "covid" in d and medical_history.get("shortness_of_breath", False)
        ]
        if treatment:
            return {"diagnosis": ", ".join(diagnoses), "treatment": treatment[0], "debug": debug_info}

        return {
            "diagnosis": ", ".join(diagnoses),
            "treatment": diagnoses[0].split(": ")[1] if diagnoses else "Tư vấn bác sĩ",
            "debug": debug_info
        }
    else:
        return {"diagnosis": "unknown", "treatment": "Tư vấn bác sĩ", "debug": debug_info}


# =======================
# Giao diện Streamlit
# =======================
st.set_page_config(page_title="Hệ Thống Y Tế Thông Minh", layout="wide")

# Chia cột
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Hệ Thống Y Tế Thông Minh")
    uploaded_file = st.file_uploader("Chọn ảnh X-ray", type=["jpg", "jpeg", "png"])

    # Hiển thị placeholder nếu chưa upload
    if not uploaded_file and os.path.exists("placeholder.jpg"):
        st.image("placeholder.jpg", caption="Chưa upload ảnh", use_container_width=True)
    elif uploaded_file:
        image_path = "temp.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Ảnh X-ray đã upload", use_container_width=True)
        if os.path.exists(image_path):
            os.remove(image_path)

with col2:
    st.write("**Triệu chứng bổ sung**")
    fatigue = st.checkbox("Có triệu chứng mệt mỏi (fatigue)")
    shortness_of_breath = st.checkbox("Có triệu chứng khó thở (shortness_of_breath)")

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

            # Kết quả chính
            st.success("Kết quả chẩn đoán:")
            st.markdown("**Chẩn đoán:** " + result['diagnosis'], unsafe_allow_html=True)
            st.markdown("**Điều trị:** " + result['treatment'], unsafe_allow_html=True)

            # Hiển thị quá trình suy luận
            with st.expander("Xem quá trình suy luận chi tiết"):
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
            st.error("Vui lòng upload ảnh X-ray trước!")
