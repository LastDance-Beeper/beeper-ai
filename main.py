import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
import json
import logging
import requests


load_dotenv()
# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 환경 변수 설정

# FastAPI 앱 생성
app = FastAPI()

# Azure OpenAI 클라이언트 생성
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# 요청 형식을 정의하는 Pydantic 모델
class Query(BaseModel):
    title: str
    content: str
    imageurl: str

class Response(BaseModel):
    title: str
    imageurl: str
    content: str
    summary: str
    recommendation: str

# 질문에 대한 답변을 생성하는 함수
def generate_answer(content: str) -> str:
    prompt = f'''너는 민원처리담당AI야
    JSON키워드는 summary와 recommendation이야
    summary에는 글이나 이미지에 대한 요약을,
    recommendation에는 글이나 이미지의 민원에 대한 해결방안을 써 
    앞에 JSON을 붙이지말고 JSON형식으로 출력해줘
    '''

    try:
        response = client.chat.completions.create(
            model= "gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.7
        )

        result = response.choices[0].message.content
        logging.info(f"Azure OpenAI Output: {result}")

        # JSON으로 파싱

        if "json\n" in result:
            result = result.replace("json\n", "").strip()
        parsed_result = json.loads(result)
        logging.info(f"Parsed JSON: {parsed_result}")
        return parsed_result

    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {str(e)}")
        raise ValueError(f"GPT 응답을 JSON으로 파싱할 수 없습니다. 응답: {result}")
    except Exception as e:
        logging.error(f"Error in generate_answer: {str(e)}")
        raise


def generate_answer_2(content: str) -> str:
    prompt = f'''너는 요약처리AI야 글을 주면 그 글을 경어체로 요약해 앞에 json을 붙이지 말고 JSON형식으로 출력해 키워드는 content야
    '''

    try:
        response = client.chat.completions.create(
            model= "gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ],
            temperature=0.5
        )

        result = response.choices[0].message.content
        logging.info(f"Azure OpenAI Output: {result}")

        # JSON으로 파싱

        if "json\n" in result:
            result = result.replace("json\n", "").strip()
        parsed_result = json.loads(result)
        logging.info(f"Parsed JSON: {parsed_result}")
        return parsed_result

    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {str(e)}")
        raise ValueError(f"GPT 응답을 JSON으로 파싱할 수 없습니다. 응답: {result}")
    except Exception as e:
        logging.error(f"Error in generate_answer: {str(e)}")
        raise

class Query1(BaseModel):
    content: str
class Response1(BaseModel):
    content: str
class ClovaSpeechClient:
    # Clova Speech invoke URL
    invoke_url = 'https://clovaspeech-gw.ncloud.com/external/v1/8762/43f0e078ac850011acc6d07dadee6b472ace926950c93c1edca6bc69ce9b6213'
    # Clova Speech secret key
    secret = '1655e22448bb424ebd1eb576c9c1899d'

    def req_url(self, url, completion, callback=None, userdata=None, forbiddens=None, boostings=None, wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                           wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   wordAlignment=True, fullText=True, diarization=None, sed=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
            'sed': sed,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        print(json.dumps(request_body, ensure_ascii=False).encode('UTF-8'))
        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body, ensure_ascii=False).encode('UTF-8'), 'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)
        return response
@app.post("/ask", response_model=Response)
async def ask_openai(query: Query):
    try:
        answer = generate_answer(query.content)

        # recommendation이 리스트일 경우 문자열로 변환
        recommendation = answer.get("recommendation", "")
        if isinstance(recommendation, list):
            recommendation = "\n".join(recommendation)

        # 각 필드를 응답으로 반환
        return {
            "title": query.title,
            "imageurl": query.imageurl,
            "content": query.content,
            "summary": answer.get("summary", ""),
            "recommendation": recommendation
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

@app.post("/text")
async def text_api(query: Query1):
    res = ClovaSpeechClient().req_upload(file='test_file.m4a', completion='sync')
    data = res.json()
    print(data["text"])
    answer = generate_answer_2(data["text"])
    json_string = {
        'content': answer["content"]
    }
    return json_string

class Query_test(BaseModel):
    content: str
class Response_test(BaseModel):
    content: str

@app.get("/test_get")
async def get_api():
    json_string = {
        'content': 'test'
    }
    return json_string

@app.post("/test_get")
async def get_api(query: Query_test):
    json_string = {
        'content': str(query.content) + 'test'
    }
    return json_string

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)


#[출처] CLOVA Speech로 Speech to text 해보기(with 로컬 환경)|작성자 소소한가
#OpenAI API