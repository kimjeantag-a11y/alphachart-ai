from PIL import Image, ImageDraw, ImageFont
import os

# 1. 파일 설정 (파일명이 정확한지 꼭 확인하세요!)
input_image_name = "독수리 심볼.jfif"  
output_image_name = "AlphaChart_Trademark.png" 
text_content = "AlphaChart AI"

# 2. 이미지 불러오기
if not os.path.exists(input_image_name):
    print(f"❌ 오류: '{input_image_name}' 파일이 없습니다. 같은 폴더에 넣어주세요.")
else:
    original_img = Image.open(input_image_name)
    
    # 3. 캔버스 설정
    base_width = 1000
    w_percent = (base_width / float(original_img.size[0]))
    h_size = int((float(original_img.size[1]) * float(w_percent)))
    img_resized = original_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
    
    padding = 50
    text_area_height = 200
    canvas_width = base_width + (padding * 2)
    canvas_height = h_size + text_area_height + (padding * 2)
    
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    
    # 4. 붙여넣기
    canvas.paste(img_resized, (padding, padding))
    
    # 5. 글씨 쓰기
    draw = ImageDraw.Draw(canvas)
    
    # 폰트 설정 (없으면 기본 폰트)
    try:
        font_path = "C:/Windows/Fonts/malgunbd.ttf" 
        font_size = 120 
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        print("⚠️ 기본 폰트로 대체합니다.")

    bbox = draw.textbbox((0, 0), text_content, font=font)
    text_width = bbox[2] - bbox[0]
    
    text_x = (canvas_width - text_width) / 2
    text_y = padding + h_size + 30 
    
    draw.text((text_x, text_y), text_content, font=font, fill="#0f172a")
    
    # 6. 저장
    canvas.save(output_image_name)
    print(f"✅ 완성! 폴더에 '{output_image_name}' 파일이 생겼습니다.")