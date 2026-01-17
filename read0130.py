import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io

# --- 影像處理核心 ---

def detect_corner_markers(img_crop_bgr):
    """辨識定位點 (A1)：尋找實心方形"""
    if img_crop_bgr.size == 0: return []
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    # 增加 block_size 提升魯棒性
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            detected_squares.append(approx.reshape(4, 2).tolist())
    return detected_squares

def detect_bubbles(img_crop_bgr):
    """
    辨識氣泡 (A2, A3)：
    加入形態學閉運算填補氣泡內字母空隙，確保圓性辨識成功。
    """
    if img_crop_bgr.size == 0: return []
    
    # 1. 預處理
    gray = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 自適應二值化 (擴大窗口以應對光影不均)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 31, 5
    )
    
    # 3. 關鍵修正：形態學閉運算 (填滿圓圈內的 A, B, C 字樣)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. 輪廓搜尋與篩選
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_circles = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 40 or peri == 0: continue
        
        # 圓性計算 formula: (4 * PI * Area) / (Perimeter^2)
        circularity = 4 * np.pi * area / (peri * peri)
        
        # 放寬篩選條件以捕捉 1-20 題可能存在的輕微形變
        if 0.5 < circularity < 1.5: 
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if 7 < radius < 35: 
                detected_circles.append([int(x), int(y), int(radius)])
                
    # 5. 排序邏輯：將辨識到的氣泡先按 X (左右欄) 再按 Y (上下) 排序
    # 假設寬度一半處為左右欄分界
    width = img_crop_bgr.shape[1]
    left_col = [c for c in detected_circles if c[0] < width / 2]
    right_col = [c for c in detected_circles if c[0] >= width / 2]
    
    left_col.sort(key=lambda c: c[1])   # 左欄由上往下
    right_col.sort(key=lambda c: c[1])  # 右欄由上往下
    
    return left_col + right_col

def draw_results_on_image(pil_image, results, region_offsets):
    """在原圖繪製辨識框與編號"""
    img_cv = np.array(pil_image.convert('RGB'))
    img_cv = img_cv[:, :, ::-1].copy() 

    for key, color, thickness in [('A1_value', (0,0,255), 4), ('A2_value', (0,255,0), 2), ('A3_value', (255,0,0), 2)]:
        if key in results:
            region = key.split('_')[0]
            off_x, off_y = region_offsets.get(region, (0, 0))
            for i, item in enumerate(results[key]):
                if key == 'A1_value':
                    pts = (np.array(item) + [off_x, off_y]).astype(np.int32)
                    cv2.polylines(img_cv, [pts], True, color, thickness)
                else:
                    cx, cy, r = item
                    # 畫出方框並標註序號
                    cv2.rectangle(img_cv, (cx + off_x - r, cy + off_y - r), (cx + off_x + r, cy + off_y + r), color, thickness)
                    if key == 'A3_value':
                        cv2.putText(img_cv, str(i+1), (cx + off_x + r, cy + off_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# --- Streamlit UI ---

st.set_page_config(page_title="答案卡辨識與校準系統 v2", layout="wide")

if 'zones' not in st.session_state:
    st.session_state.update({
        'img_file': None, 'original_image': None, 'resized_image': None, 
        'scale_factor': 1.0, 'zones': {'A1': None, 'A2': None, 'A3': None, 'A4': None},
        'cropping_mode': None, 'temp_box': None, 'recognition_results': None, 'result_image': None
    })

st.title("🎯 答案卡精確辨識系統 (已優化 1-20 題)")

col_left, col_right = st.columns([1.5, 2.5])

with col_left:
    st.header("1. 檔案上傳")
    uploaded_file = st.file_uploader("上傳答案卡圖片", type=["jpg", "png", "jpeg"])

    if uploaded_file and st.session_state.img_file != uploaded_file:
        st.session_state.img_file = uploaded_file
        st.session_state.original_image = Image.open(uploaded_file)
        # 預覽縮放
        display_width = 850
        orig_w, orig_h = st.session_state.original_image.size
        w_ratio = display_width / orig_w
        st.session_state.resized_image = st.session_state.original_image.resize((display_width, int(orig_h * w_ratio)), Image.LANCZOS)
        st.session_state.scale_factor = 1 / w_ratio
        st.session_state.zones = {k: None for k in st.session_state.zones}
        st.session_state.recognition_results = None
        st.session_state.result_image = None

    st.markdown("### 2. 區域校準")
    for zone in ['A1', 'A2', 'A3', 'A4']:
        label = {"A1":"定位點","A2":"基本資料","A3":"選擇題","A4":"手寫區"}[zone]
        c_btn, c_status, c_ok = st.columns([2, 0.5, 1])
        
        is_active = st.session_state.cropping_mode == zone
        c_btn.button(f"標示 {zone} {label}", key=f"btn_{zone}", 
                     type="primary" if is_active else "secondary", use_container_width=True,
                     on_click=lambda z=zone: st.session_state.update({"cropping_mode": z}))
        
        if st.session_state.zones[zone]: c_status.markdown("✅")
        
        if is_active and c_ok.button("確定", key=f"ok_{zone}", type="primary", use_container_width=True):
            st.session_state.zones[zone] = st.session_state.temp_box
            st.session_state.cropping_mode = None
            st.rerun()

    st.divider()
    if st.button("🚀 執行辨識", type="primary", use_container_width=True, disabled=not all(st.session_state.zones.values())):
        with st.spinner("辨識引擎運算中..."):
            full_cv = cv2.cvtColor(np.array(st.session_state.original_image.convert('RGB')), cv2.COLOR_RGB2BGR)
            results, offsets = {}, {}
            scale = st.session_state.scale_factor
            
            for z in ['A1', 'A2', 'A3']:
                box = st.session_state.zones[z]
                rx, ry, rw, rh = int(box['left']*scale), int(box['top']*scale), int(box['width']*scale), int(box['height']*scale)
                crop = full_cv[ry:ry+rh, rx:rx+rw]
                offsets[z] = (rx, ry)
                results[f"{z}_value"] = detect_corner_markers(crop) if z=='A1' else detect_bubbles(crop)
            
            st.session_state.result_image = draw_results_on_image(st.session_state.original_image, results, offsets)
            st.session_state.recognition_results = results
            st.success(f"完成！共偵測到 {len(results.get('A3_value', []))} 個選擇題氣泡。")

with col_right:
    if st.session_state.resized_image:
        mode = st.session_state.cropping_mode
        if mode:
            img_w, _ = st.session_state.resized_image.size
            st.info(f"請在地圖上拖曳藍框以覆蓋「{mode}」區域")
            box_data = st_cropper(st.session_state.resized_image, realtime_update=True, box_color='#0000FF', 
                                  aspect_ratio=None, return_type='box', key=f"crop_{mode}")
            st.session_state.temp_box = box_data
        elif st.session_state.result_image:
            st.image(st.session_state.result_image, caption="辨識結果：已自動排除非氣泡噪點並排序題號")
            # 顯示偵測到的座標數據
            with st.expander("查看原始座標數據"):
                st.write(st.session_state.recognition_results)
        else:
            st.image(st.session_state.resized_image, caption="原始檔案預覽")
