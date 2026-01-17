import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# --- 核心辨識引擎：強化版 ---

def detect_bubbles(img_crop_bgr):
    """
    極限強化版氣泡辨識：
    針對內部有字母的圓圈優化，確保 1-20 題不再漏抓。
    """
    if img_crop_bgr.size == 0: return []
    
    # 1. 預處理：轉灰階 + 模糊
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 2. 強力二值化：讓黑白對比更極端
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 35, 10
    )
    
    # 3. 關鍵修正：形態學「閉運算」+「膨脹」
    # 使用較大的圓形核，強制把圓圈內的 A, B, C 字母連在一起變成實心圓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    # 4. 輪廓搜尋
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_circles = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 60 or peri == 0: continue
        
        # 圓性計算
        circularity = 4 * np.pi * area / (peri * peri)
        
        # 邊界框計算
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # 放寬標準：只要夠圓且長寬比接近 1:1 就納入
        if 0.4 < circularity < 1.6 and 0.7 < aspect_ratio < 1.3:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            # 依據常見答案卡比例，半徑通常在 8-30 像素之間
            if 8 < radius < 40:
                detected_circles.append([int(cx), int(cy), int(radius)])
                
    # 5. 排序邏輯：解決題號混亂問題
    if not detected_circles: return []
    
    width = img_crop_bgr.shape[1]
    # 將氣泡分為左、右兩半（1-20題 vs 21-40題）
    left_col = [c for c in detected_circles if c[0] < width * 0.5]
    right_col = [c for c in detected_circles if c[0] >= width * 0.5]
    
    # 分別由上往下排序
    left_col.sort(key=lambda c: c[1])
    right_col.sort(key=lambda c: c[1])
    
    return left_col + right_col

def detect_corner_markers(img_crop_bgr):
    """定位點辨識 (A1)"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                squares.append(approx.reshape(4, 2).tolist())
    return squares

def draw_results(pil_image, results, offsets):
    """在圖片上標註題號，方便檢查是否有跳號"""
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # 繪製選擇題 (A3)
    if 'A3_value' in results:
        off_x, off_y = offsets.get('A3', (0, 0))
        for i, (cx, cy, r) in enumerate(results['A3_value']):
            # 畫圓圈
            cv2.circle(img_cv, (cx + off_x, cy + off_y), r, (0, 0, 255), 2)
            # 標題號
            cv2.putText(img_cv, str(i+1), (cx + off_x - 10, cy + off_y - r - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# --- Streamlit 介面 ---

st.set_page_config(page_title="AI 答案卡校正系統 Pro", layout="wide")

# 初始化狀態
if 'zones' not in st.session_state:
    st.session_state.update({'img': None, 'zones': {'A1':None,'A2':None,'A3':None,'A4':None}, 'mode': None})

st.title("🎯 答案卡精準辨識系統 (強化版)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. 上傳與設定")
    up = st.file_uploader("上傳答案卡影像", type=['jpg','png','jpeg'])
    if up:
        st.session_state.img = Image.open(up)
        
    for z in ['A1', 'A2', 'A3', 'A4']:
        name = {"A1":"定位點","A2":"基本資料","A3":"選擇題","A4":"手寫區"}[z]
        c_btn, c_ok = st.columns([2, 1])
        if c_btn.button(f"標示 {name}", use_container_width=True):
            st.session_state.mode = z
        if st.session_state.zones[z]:
            c_ok.write("✅")

    if st.button("🚀 開始辨識", type="primary", use_container_width=True):
        if all(st.session_state.zones.values()):
            orig = st.session_state.img
            orig_cv = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
            w_ratio = orig.size[0] / 850 # 假設預覽寬度 850
            
            res_data, off_data = {}, {}
            for z in ['A1', 'A2', 'A3']:
                b = st.session_state.zones[z]
                # 換算回原始尺寸
                rx, ry, rw, rh = [int(v * w_ratio) for v in [b['left'], b['top'], b['width'], b['height']]]
                crop = orig_cv[ry:ry+rh, rx:rx+rw]
                off_data[z] = (rx, ry)
                res_data[f"{z}_value"] = detect_corner_markers(crop) if z=='A1' else detect_bubbles(crop)
            
            st.session_state.res_img = draw_results(orig, res_data, off_data)
            st.success("辨識完成！")

with col2:
    if st.session_state.img:
        if st.session_state.mode:
            st.info(f"請在下方圖片選取【{st.session_state.mode}】區域，選完按下方確定")
            # 限制預覽寬度
            preview = st.session_state.img.resize((850, int(850 * st.session_state.img.size[1]/st.session_state.img.size[0])))
            box = st_cropper(preview, realtime_update=True, box_color='blue', aspect_ratio=None, return_type='box')
            if st.button("確定選取"):
                st.session_state.zones[st.session_state.mode] = box
                st.session_state.mode = None
                st.rerun()
        elif 'res_img' in st.session_state:
            st.image(st.session_state.res_img, caption="辨識結果（附帶題號檢查）")
        else:
            st.image(st.session_state.img, use_container_width=True)
