% ==========================
% Luat suy luan benh
% ==========================

% Cac luat chan doan (dua vao trieu chung)
chan_doan(viem_phoi) :-
    trieu_chung(ho),
    trieu_chung(sot),
    trieu_chung(dau_nguc).

chan_doan(cum) :-
    trieu_chung(sot),
    trieu_chung(met_moi).

chan_doan(covid) :-
    trieu_chung(sot),
    trieu_chung(kho_tho).

% Binh thuong khi ML du doan anh la binh thuong
chan_doan(binh_thuong) :-
    trieu_chung(coi_la_binh_thuong).

% Neu khong khop benh nao → khong_xac_dinh
chan_doan(khong_xac_dinh) :-
    \+ chan_doan(viem_phoi),
    \+ chan_doan(cum),
    \+ chan_doan(covid),
    \+ chan_doan(binh_thuong).

% --------------------------
% Cac luat dieu tri
% --------------------------
dieu_tri(viem_phoi, "Kháng sinh như amoxicillin, nghỉ ngơi, uống nhiều nước").
dieu_tri(cum, "Thuốc kháng virus như Tamiflu, nghỉ ngơi, bù nước").
dieu_tri(covid, "Tư vấn bác sĩ, xét nghiệm PCR, cách ly").
dieu_tri(binh_thuong, "Phổi bình thường, không cần điều trị").
dieu_tri(khong_xac_dinh, "Tư vấn bác sĩ để được chẩn đoán chính xác hơn").