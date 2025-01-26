from flask import Flask, request, render_template
from keras import models
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
import wikipediaapi
import base64
from io import BytesIO

app = Flask(__name__)

# 학습된 모델 로드
model = models.load_model('model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 클래스 이름 (영문) 매핑
class_names = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet',
    'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout'
]

# 영문 -> 한글 매핑
english_to_korean = {
    'Black Sea Sprat': '흑해 멸치',
    'Gilt-Head Bream': '참돔',
    'Horse Mackerel': '전갱이',
    'Red Mullet': '붉은 숭어',
    'Red Sea Bream': '황돔',
    'Sea Bass': '농어',
    'Shrimp': '새우',
    'Striped Red Mullet': '줄무늬 붉은 숭어',
    'Trout': '송어',

}


def get_wikipedia_summary_and_image(search_term):
    user_agent = "MyImageAnalyzerApp/1.0 (https://example.com; contact@example.com)"
    wiki_wiki = wikipediaapi.Wikipedia(language='ko', user_agent=user_agent)
    page = wiki_wiki.page(search_term)

    if page.exists():
        summary = page.summary[:300]  # 요약 정보 최대 300자
        image_url = None
        try:
            response = requests.get(page.fullurl)
            soup = BeautifulSoup(response.text, 'html.parser')
            image = soup.find('img')  # 첫 번째 이미지 가져오기
            if image:
                image_url = f"https:{image['src']}"
        except Exception as e:
            print(f"이미지 가져오는 중 오류 발생: {e}")
        return summary, image_url
    else:
        return "위키백과 페이지를 찾을 수 없습니다.", None

# 영어 클래스 이름을 한글로 매핑 후 위키백과에서 검색
def search_wikipedia_by_prediction(predicted_class):
    # 영어 클래스 이름을 한글로 매핑
    korean_name = english_to_korean.get(predicted_class, predicted_class)
    # 위키백과 검색 수행
    summary, image_url = get_wikipedia_summary_and_image(korean_name)
    return korean_name, summary, image_url

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis', methods=['POST'])
def analysis():
    if 'file' not in request.files:
        return "이미지 파일이 업로드되지 않았습니다."

    file = request.files['file']
    if file.filename == '':
        return "파일이 선택되지 않았습니다."

    try:
        img = Image.open(file).convert('RGB').resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        predicted_class_english = class_names[class_index]
        predicted_class_korean = english_to_korean.get(predicted_class_english, "알 수 없음")

        summary, image_url = get_wikipedia_summary_and_image(predicted_class_english)

        # 이미지를 Base64로 변환
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    except Exception as e:
        print(f"Error during processing: {e}")
        return "이미지 처리 중 오류가 발생했습니다."

    return render_template(
        'analysis.html',
        result_korean=predicted_class_korean,
        result_english=predicted_class_english,
        wiki_description=summary,
        wiki_image_url=image_url,
        uploaded_image=f"data:image/png;base64,{img_base64}"
    )

if __name__ == '__main__':
    app.run(debug=True)