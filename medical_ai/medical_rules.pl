% ==========================
% Luật suy luận bệnh
% ==========================

% Các luật chẩn đoán (dựa vào triệu chứng)
diagnosis(pneumonia) :-
    current_symptom(cough),
    current_symptom(fever),
    current_symptom(chest_pain).

diagnosis(flu) :-
    current_symptom(fever),
    current_symptom(fatigue).

diagnosis(covid) :-
    current_symptom(fever),
    current_symptom(shortness_of_breath).

% Normal khi ML dự đoán ảnh là bình thường
diagnosis(normal) :-
    current_symptom(normal_flag).

% Nếu không khớp bệnh nào → unknown
diagnosis(unknown) :-
    \+ diagnosis(pneumonia),
    \+ diagnosis(flu),
    \+ diagnosis(covid),
    \+ diagnosis(normal).

% --------------------------
% Các luật điều trị
% --------------------------
treatment(pneumonia, "Kháng sinh như amoxicillin, nghỉ ngơi, uống nhiều nước").
treatment(flu, "Thuốc kháng virus như Tamiflu, nghỉ ngơi, bù nước").
treatment(covid, "Tư vấn bác sĩ, xét nghiệm PCR, cách ly").
treatment(normal, "Phổi bình thường, không cần điều trị").
treatment(unknown, "Tư vấn bác sĩ để được chẩn đoán chính xác hơn").
