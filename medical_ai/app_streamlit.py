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
        debug_info.append("Đã load medical_rules.pl thành công")
    except Exception as e:
        debug_info.append(f"Lỗi khi load medical_rules.pl: {str(e)}")
        return {"chan_doan": "unknown", "dieu_tri": f"Lỗi Prolog: {str(e)}", "debug": debug_info}

    # Kiểm tra triệu chứng trước retractall
    current_syms = list(prolog.query("current_symptom(X)"))
    debug_info.append(f"Triệu chứng trước retractall: {current_syms}")

    # Xóa tất cả các triệu chứng hiện tại
    try:
        prolog.query("retractall(current_symptom(_))")
        debug_info.append("Đã xóa tất cả các triệu chứng hiện tại trong Prolog")
        current_syms = list(prolog.query("current_symptom(X)"))
        debug_info.append(f"Triệu chứng sau retractall: {current_syms}")
        if current_syms:
            debug_info.append("Cảnh báo: Triệu chứng vẫn tồn tại sau retractall! Bỏ qua triệu chứng hiện tại...")
            prolog = Prolog()  # Khởi tạo lại Prolog
            prolog.consult("medical_rules.pl")
            current_syms = list(prolog.query("current_symptom(X)"))
            debug_info.append(f"Triệu chứng sau khi khởi tạo lại Prolog: {current_syms}")
            if current_syms:
                debug_info.append("Lỗi nghiêm trọng: Triệu chứng vẫn tồn tại sau khởi tạo lại Prolog! Bỏ qua Prolog hiện tại.")
                # Chỉ sử dụng triệu chứng từ symptoms và medical_history
                expected_syms = set(symptoms)
                if medical_history.get("fatigue", False):
                    expected_syms.add("fatigue")
                if medical_history.get("shortness_of_breath", False):
                    expected_syms.add("shortness_of_breath")
                if diagnosis == "NORMAL":
                    expected_syms.add("normal_flag")
                debug_info.append(f"Sử dụng triệu chứng dự kiến: {expected_syms}")
    except Exception as e:
        debug_info.append(f"Lỗi khi xóa triệu chứng bằng retractall: {str(e)}")
        return {"chan_doan": "unknown", "dieu_tri": "Lỗi Prolog", "debug": debug_info}

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

    # Chỉ thêm normal_flag nếu diagnosis là NORMAL
    if diagnosis == "NORMAL":
        prolog.assertz("current_symptom(normal_flag)")
        debug_info.append("Đã thêm: current_symptom(normal_flag)")

    # Query chẩn đoán và điều trị
    try:
        results = list(prolog.query("diagnosis(D), treatment(D, T)"))
    except Exception as e:
        debug_info.append(f"Lỗi query Prolog: {str(e)}")
        return {"chan_doan": "unknown", "dieu_tri": "Tư vấn bác sĩ", "debug": debug_info}

    # Loại bỏ triệu chứng trùng lặp
    current_syms = list({s['X'] for s in prolog.query("current_symptom(X)")})
    debug_info.append(f"Các triệu chứng hiện tại trong Prolog (không trùng lặp): {current_syms}")

    # Nếu có lỗi trước đó, sử dụng triệu chứng dự kiến
    if "Lỗi nghiêm trọng" in " ".join(debug_info):
        current_syms = list(expected_syms)
        debug_info.append(f"Đã sử dụng triệu chứng dự kiến thay thế: {current_syms}")

    if results:
        diagnoses = []
        seen = set()
        ml_diagnosis = "pneumonia" if diagnosis == "PNEUMONIA" else "normal"
        
        # Lọc chẩn đoán dựa trên triệu chứng bổ sung và mô hình AI
        for result in results:
            diag = safe_decode(result['D'])
            treat = safe_decode(result['T'])
            if (diag == ml_diagnosis or
                (diag == "flu" and medical_history.get("fatigue", False) and "fever" in current_syms) or
                (diag == "covid" and medical_history.get("shortness_of_breath", False) and "fever" in current_syms)):
                if diag not in seen:
                    diagnoses.append(f"{diag}: {treat}")
                    seen.add(diag)

        # Ưu tiên điều trị
        treatment = [
            d.split(": ")[1]
            for d in diagnoses
            if "covid" in d and medical_history.get("shortness_of_breath", False)
        ]
        if not treatment:
            treatment = [
                d.split(": ")[1]
                for d in diagnoses
                if ml_diagnosis in d
            ]
        if not treatment:
            treatment = [diagnoses[0].split(": ")[1]] if diagnoses else ["Tư vấn bác sĩ"]

        return {
            "chan_doan": ", ".join(diagnoses) if diagnoses else "unknown",
            "dieu_tri": treatment[0],
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