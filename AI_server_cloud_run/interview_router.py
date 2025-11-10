"""
Phase 2: 면접 진행 라우터
면접 시작 → 영상 업로드 → STT → 세션 Agent → 다음 질문
"""

import os
import json
import logging
import base64
import re
from datetime import datetime
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from google.cloud import storage
from google.cloud import texttospeech

logger = logging.getLogger(__name__)

# 환경 설정 - 환경변수에서 로드 (기본값 없음)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
SESSION_AGENT_ID = os.getenv("SESSION_AGENT_ID")

if not all([PROJECT_ID, BUCKET_NAME, SESSION_AGENT_ID]):
    raise ValueError(
        "환경변수를 .env 파일에 설정해주세요: "
        "GOOGLE_CLOUD_PROJECT, GCS_BUCKET_NAME, SESSION_AGENT_ID"
    )

router = APIRouter(prefix="/api/interview", tags=["면접 진행"])

# ✅ ADK 세션 ID 저장소 (비즈니스 세션 ID → ADK 세션 ID 매핑)
# 예: {"session_20251107_183601": "adk_session_abc123"}
adk_session_store = {}


# ========== Response 모델 ==========

class InterviewStartResponse(BaseModel):
    """면접 시작 응답"""
    status: str
    questionId: int
    question: str
    isTailQuestion: bool
    sessionId: str
    remainingSlots: int
    audioData: Optional[str] = None  # Base64 인코딩된 오디오 (MP3)


class InterviewAnswerResponse(BaseModel):
    """답변 제출 응답"""
    status: str  # "continue" | "completed"
    questionId: Optional[int] = None
    question: Optional[str] = None
    isTailQuestion: Optional[bool] = None
    sessionId: str
    remainingSlots: int
    message: Optional[str] = None  # status="completed"일 때
    audioData: Optional[str] = None  # Base64 인코딩된 오디오 (MP3)


# ========== Helper Functions ==========

async def call_session_agent(message: str, session_id: Optional[str] = None, is_first_call: bool = False) -> dict:
    """세션 관리 Agent 호출
    
    Args:
        message: Agent에게 전달할 메시지 ([SESSION_ID: xxx] 포함)
        session_id: 비즈니스 세션 ID (파일명용)
        is_first_call: 첫 호출 여부 (True면 새 ADK 세션 생성)
    """
    try:
        # ✅ 비즈니스 세션 ID 추출 (파일명용)
        business_session_id = session_id
        
        if not business_session_id:
            # 메시지에서 세션 ID 추출 시도
            session_match = re.search(r'\[SESSION_ID:\s*(\S+)\]', message)
            business_session_id = session_match.group(1) if session_match else None
        
        if not business_session_id:
            logger.warning("⚠️ 세션 ID를 찾을 수 없습니다. 기본값 사용")
            business_session_id = "default_session"
        
        logger.info(f"🆔 비즈니스 세션 ID: {business_session_id}")
        
        client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
        # 프로젝트 ID를 사용 (프로젝트 넘버는 자동으로 해석됨)
        agent_resource_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{SESSION_AGENT_ID}"
        adk_app = client.agent_engines.get(name=agent_resource_name)
        
        # ✅ ADK 세션 관리: 첫 호출이면 생성, 아니면 재사용
        if is_first_call:
            # 첫 호출: ADK 세션 생성
            logger.info(f"🆕 새 ADK 세션 생성 중...")
            adk_session = await adk_app.async_create_session(user_id="interview_user")
            adk_session_id = adk_session.get("id")
            adk_session_store[business_session_id] = adk_session_id
            logger.info(f"✅ ADK 세션 생성 완료: {adk_session_id}")
        else:
            # 이후 호출: 저장된 ADK 세션 ID 사용
            adk_session_id = adk_session_store.get(business_session_id)
            if not adk_session_id:
                raise ValueError(f"저장된 ADK 세션 ID를 찾을 수 없습니다: {business_session_id}")
            logger.info(f"♻️ 기존 ADK 세션 재사용: {adk_session_id}")
        
        events = []
        
        # ✅ ADK 세션 ID를 명시하여 세션 유지!
        # - user_id: 고정값
        # - session_id: ADK가 생성한 세션 ID (컨텍스트 유지)
        # - message: [SESSION_ID: xxx] 포함 (파일명 매칭용)
        async for event in adk_app.async_stream_query(
            user_id="interview_user",  # 고정
            session_id=adk_session_id,  # ✅ ADK 세션 ID 사용!
            message=message  # [SESSION_ID: xxx] 포함
        ):
            events.append(event)
            logger.info(f"📦 Event #{len(events)}: {type(event).__name__}")
        
        if not events:
            raise ValueError("Agent 응답이 없습니다.")
        
        # 마지막 이벤트 확인
        last_event = events[-1]
        logger.info(f"🎯 Last Event Type: {type(last_event)}")
        
        # JSON 파싱 시도
        try:
            # 1. Dict에서 텍스트 추출
            if isinstance(last_event, dict):
                text = last_event.get('content', {}).get('parts', [{}])[0].get('text', '')
                logger.info(f"📝 Extracted text: {text[:200]}...")
            elif isinstance(last_event, str):
                text = last_event
            else:
                text = str(last_event)
            
            # 2. 마크다운 코드 블록 제거 (```json ... ```)
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("✅ Markdown code block removed")
            else:
                json_str = text
                logger.info("⚠️ No markdown block found, using text as-is")
            
            # 3. JSON 파싱
            response = json.loads(json_str)
            logger.info(f"✅ Parsed Response Keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {str(e)}")
            logger.error(f"Raw text: {text[:500]}")
            raise ValueError(f"Agent 응답을 JSON으로 파싱할 수 없습니다: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Parsing error: {str(e)}")
            raise ValueError(f"Agent 응답 파싱 실패: {str(e)}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Agent 호출 실패: {str(e)}")
        raise


def fix_pronunciation(text: str) -> str:
    """약어 및 고유명사를 올바른 발음으로 변환
    
    Args:
        text: 원본 텍스트
    
    Returns:
        발음이 수정된 텍스트
    """
    # 대학/기관 약어
    replacements = {
        "KAIST": "카이스트",
        "kaist": "카이스트",
        "Kaist": "카이스트",
        "KIST": "키스트",
        "ETRI": "이티알아이",
        "MIT": "엠아이티",
        
        # 기업명
        "LIG Nex1": "엘아이지 넥스원",
        "LIG넥스원": "엘아이지 넥스원",
        "LIGNex1": "엘아이지 넥스원",
        "LIG": "엘아이지",
        
        # 기술 약어
        "ROS": "로스",
        "AI": "에이아이",
        "ML": "엠엘",
        "GPS": "지피에스",
        "SLAM": "슬램",
        "IMU": "아이엠유",
        "UAV": "유에이비",
        "UGV": "유지비",
        "IoT": "아이오티",
        
        # 프로그래밍 언어
        "Python": "파이썬",
        "C++": "씨플플",
        "C#": "씨샵",
    }
    
    result = text
    for original, pronunciation in replacements.items():
        result = result.replace(original, pronunciation)
    
    return result


def enhance_question_with_markup(text: str, is_tail_question: bool) -> str:
    """질문 텍스트에 마크업 태그를 자동으로 추가하여 자연스러운 면접관 억양 생성
    
    Args:
        text: 원본 질문 텍스트
        is_tail_question: 꼬리질문 여부
    
    Returns:
        마크업 태그가 추가된 텍스트
    """
    # 0. 약어 및 고유명사 발음 수정
    enhanced = fix_pronunciation(text)
    
    # 1. 인사말 뒤에 중간 일시중지 (처음 만났을 때의 자연스러운 쉼)
    greetings = ["안녕하세요", "반갑습니다", "좋습니다"]
    for greeting in greetings:
        if f"{greeting}." in enhanced or f"{greeting}," in enhanced:
            enhanced = enhanced.replace(f"{greeting}.", f"{greeting}.[long pause]")
            enhanced = enhanced.replace(f"{greeting},", f"{greeting},[medium pause]")
    
    # 2. 서류/자기소개서 언급 뒤 쉼 (자료를 본 느낌)
    review_phrases = ["서류를 보니", "자기소개서를 보니", "이력서를 보니"]
    for phrase in review_phrases:
        if phrase in enhanced:
            enhanced = enhanced.replace(phrase, f"{phrase}[medium pause]")
    
    # 3. 중요한 접속사 뒤에 짧은 일시중지 (생각 정리 시간)
    connectors = ["또한", "그리고", "특히", "예를 들어", "더불어", "아울러"]
    for connector in connectors:
        if f"{connector}," in enhanced:
            enhanced = enhanced.replace(f"{connector},", f"{connector}[short pause],")
    
    # 4. 긴 문장 중간에 쉼표 뒤 쉼 추가 (자연스러운 호흡)
    enhanced = enhanced.replace(",", ",[short pause]")
    
    # 5. "본인이", "귀하께서" 같은 존칭 뒤 짧은 쉼
    honorifics = ["본인이", "본인의", "귀하께서는", "귀하의"]
    for honorific in honorifics:
        if honorific in enhanced and f"{honorific}[short pause]" not in enhanced:
            enhanced = enhanced.replace(honorific, f"{honorific}[short pause]")
    
    # 6. 질문의 핵심 키워드 앞에 짧은 일시중지 (강조)
    key_phrases = ["어떻게", "무엇을", "왜", "어떤", "어느", "어디에", "언제"]
    for phrase in key_phrases:
        if phrase in enhanced and f"[short pause]{phrase}" not in enhanced:
            enhanced = enhanced.replace(f" {phrase}", f" [short pause]{phrase}", 1)
    
    # 7. 중요한 명사 앞에 쉼 (강조)
    important_words = ["역량", "경험", "포부", "비전", "목표", "성과"]
    for word in important_words:
        if f"'{word}" in enhanced or f"'{word}" in enhanced:
            enhanced = enhanced.replace(f"'{word}", f"'[short pause]{word}")
            enhanced = enhanced.replace(f"'{word}", f"'[short pause]{word}")
    
    # 8. 질문 끝 전에 짧은 일시중지 (답변 준비 시간)
    if "?" in enhanced:
        enhanced = enhanced.replace("요?", "요[medium pause]?")
        enhanced = enhanced.replace("까?", "까[medium pause]?")
        enhanced = enhanced.replace("가?", "가[medium pause]?")
    
    # 9. 꼬리질문은 호기심있는 어조
    if is_tail_question:
        if "방금" in enhanced:
            enhanced = enhanced.replace("방금", "[curious]방금")
        if "그렇다면" in enhanced:
            enhanced = enhanced.replace("그렇다면", "[curious]그렇다면")
    
    logger.debug(f"🎨 마크업 적용: {text[:30]}... → {enhanced[:50]}...")
    return enhanced


def text_to_speech(text: str, is_tail_question: bool = False) -> str:
    """텍스트를 음성으로 변환하고 Base64로 인코딩하여 반환 (Gemini-TTS + 면접관 스타일)
    
    Args:
        text: 변환할 텍스트 (면접 질문)
        is_tail_question: 꼬리질문 여부 (True: 꼬리질문, False: 메인 질문)
    
    Returns:
        Base64 인코딩된 MP3 오디오 데이터
    """
    try:
        # ✅ 질문 길이 체크 (너무 길면 잘라내기)
        MAX_TTS_LENGTH = 500  # 최대 500자로 제한 (타임아웃 방지)
        if len(text) > MAX_TTS_LENGTH:
            logger.warning(f"⚠️ 질문이 너무 깁니다 ({len(text)}자). {MAX_TTS_LENGTH}자로 제한합니다.")
            text = text[:MAX_TTS_LENGTH] + "..."
        
        logger.info(f"🎤 TTS 시작 (면접관 스타일): {text[:50]}... [{'꼬리질문' if is_tail_question else '메인질문'}] ({len(text)}자)")
        
        # TTS 클라이언트 생성
        tts_client = texttospeech.TextToSpeechClient()
        
        # 🎨 마크업 태그 자동 삽입 (자연스러운 쉼과 억양)
        enhanced_text = enhance_question_with_markup(text, is_tail_question)
        
        # 입력 텍스트 설정 (마크업 태그만 사용)
        # Note: style_prompt는 SynthesisInput에서 지원하지 않음
        # 대신 마크업 태그 + 오디오 파라미터로 면접관 스타일 구현
        synthesis_input = texttospeech.SynthesisInput(
            text=enhanced_text  # ✅ 마크업 태그가 포함된 텍스트
        )
        
        # ✅ Gemini-TTS 음성 설정 (Laomedeia - 자연스러운 여성 음성)
        # 공식 문서: https://cloud.google.com/text-to-speech/docs/gemini-tts
        voice = texttospeech.VoiceSelectionParams(
            name="Laomedeia",                 # Gemini-TTS 음성 이름
            model_name="gemini-2.5-pro-tts",  # ✅ Gemini-TTS 모델 지정 (필수!)
            language_code="ko-KR"             # 한국어
        )
        
        # 오디오 설정 (면접관 스타일 최적화)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # 더 느리게 (면접관의 신중하고 명확한 말투)
            pitch=-5.0,          # 낮게 (권위있고 안정적인 톤)
        )
        
        # TTS 실행 (Gemini-TTS)
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Base64 인코딩
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        
        logger.info(f"✅ TTS 완료: {len(audio_base64)} chars (Base64) [면접관 스타일 적용]")
        return audio_base64
        
    except Exception as e:
        logger.error(f"❌ TTS 실패: {str(e)}")
        # TTS 실패 시 None 반환 (텍스트만 전달)
        return None


async def video_to_text(video_uri: str) -> str:
    """Gemini 2.5 Flash로 영상 → 텍스트 변환 (STT)"""
    try:
        logger.info(f"🎤 STT 시작: {video_uri}")
        
        # Gemini 2.5 Flash 모델
        model = GenerativeModel("gemini-2.5-flash")
        
        # 영상 확장자 확인하여 mime_type 결정
        if video_uri.endswith('.webm'):
            mime_type = "video/webm"
        elif video_uri.endswith('.mp4'):
            mime_type = "video/mp4"
        else:
            mime_type = "video/webm"  # 기본값
        
        logger.info(f"📹 Video MIME type: {mime_type}")
        
        # 영상 파트 생성
        video_part = Part.from_uri(video_uri, mime_type=mime_type)
        
        # STT 프롬프트
        prompt = """이 영상에서 사람이 말하는 내용을 정확하게 텍스트로 변환해주세요.

규칙:
- 음성만 텍스트로 변환
- 배경 소음은 무시
- 문장 부호 자동 추가
- 자연스러운 문장으로 정리

변환된 텍스트만 출력하세요."""
        
        # 요청 전송
        response = model.generate_content([prompt, video_part])
        
        text_result = response.text.strip()
        
        # Note: 로그는 호출하는 쪽(upload_answer)에서 찍음
        return text_result
        
    except Exception as e:
        logger.error(f"❌ STT 실패: {str(e)}")
        raise ValueError(f"영상 텍스트 변환 실패: {str(e)}")


def update_progress_video_url(session_id: str, question_number: int, video_url: str):
    """Progress JSON에 videoUrl 업데이트"""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        progress_path = f"progress_interview/{session_id}_progress.json"
        blob = bucket.blob(progress_path)
        
        if not blob.exists():
            logger.warning(f"⚠️ Progress 파일 없음: {progress_path}")
            return
        
        # 기존 데이터 로드
        progress_data = json.loads(blob.download_as_text())
        
        # 해당 질문 찾기
        questions = progress_data.get("questions", [])
        for q in questions:
            if q.get("number") == question_number:  # ✅ session_agent가 사용하는 "number" 키
                q["videoUrl"] = video_url
                q["uploadedAt"] = datetime.now().isoformat()
                logger.info(f"✅ videoUrl 업데이트: Q{question_number}")
                break
        
        # 저장
        blob.upload_from_string(
            json.dumps(progress_data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"❌ Progress 업데이트 실패: {str(e)}")


# ========== 엔드포인트 ==========

@router.post("/start", response_model=InterviewStartResponse)
async def start_interview(sessionId: str = Form(...)):
    """
    면접 시작
    
    **Process:**
    1. ✅ Phase 1에서 받은 세션 ID 사용
    2. 세션 ID를 메시지에 포함하여 Agent 호출
    3. Agent가 분석 파일 로드 (session_id_analysis.json)
    4. 첫 번째 질문 생성 및 반환
    
    **Parameters:**
    - sessionId: Phase 1에서 받은 세션 ID (필수!)
    
    **Returns:**
    - status: "continue"
    - questionId: 질문 ID
    - question: 질문 텍스트
    - isTailQuestion: false (첫 질문은 항상 메인 질문)
    - sessionId: 입력받은 세션 ID
    - remainingSlots: 남은 질문 개수
    """
    
    try:
        logger.info("🎬 면접 시작 요청")
        logger.info(f"🆔 세션 ID 수신: {sessionId}")
        
        # ✅ 첫 호출: [SESSION_ID: xxx] 포함 (분석 데이터 로드)
        message_with_session = f"면접을 시작하겠습니다. 첫 번째 질문을 주세요.\n[SESSION_ID: {sessionId}]"
        
        # ✅ Agent 호출 (첫 호출이므로 ADK 세션 생성)
        response = await call_session_agent(message_with_session, session_id=sessionId, is_first_call=True)
        
        logger.info(f"🔍 Response Keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        logger.info(f"🔍 Full Response: {json.dumps(response, ensure_ascii=False, indent=2)}")
        
        logger.info(f"✅ 면접 시작: {sessionId}")
        
        # ✅ TTS: 질문을 음성으로 변환 (면접관 스타일)
        question_text = response.get("question", "")
        is_tail = response.get("isTailQuestion", False)
        audio_data = None
        if question_text:
            audio_data = text_to_speech(question_text, is_tail_question=is_tail)
        
        return InterviewStartResponse(
            status=response.get("status", "continue"),
            questionId=response.get("questionId"),
            question=question_text,
            isTailQuestion=is_tail,
            sessionId=sessionId,  # ✅ Phase 1에서 받은 sessionId 반환
            remainingSlots=response.get("remainingSlots", 0),
            audioData=audio_data  # ✅ Base64 인코딩된 오디오 (면접관 스타일)
        )
        
    except Exception as e:
        logger.error(f"❌ 면접 시작 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"면접 시작 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/upload-answer", response_model=InterviewAnswerResponse)
async def upload_answer(
    sessionId: str = Form(...),
    questionNumber: int = Form(...),
    videoFile: UploadFile = File(...)
):
    """
    답변 영상 업로드 및 다음 질문 요청
    
    **Process:**
    1. 영상 파일 GCS 업로드
    2. Progress JSON에 videoUrl 업데이트
    3. Gemini 2.5 Flash로 STT (영상 → 텍스트)
    4. 세션 Agent에 텍스트 전달
    5. 다음 질문 또는 종료 신호 받기
    
    **Parameters:**
    - sessionId: 세션 ID (start에서 받은 값)
    - questionNumber: 현재 질문 번호
    - videoFile: 녹화된 영상 파일 (webm 또는 mp4)
    
    **Returns:**
    - status: "continue" | "completed"
    - questionId: 다음 질문 ID (continue일 때)
    - question: 다음 질문 텍스트 (continue일 때)
    - isTailQuestion: 꼬리질문 여부
    - sessionId: 세션 ID
    - remainingSlots: 남은 질문 개수
    - message: 종료 메시지 (completed일 때)
    """
    
    try:
        logger.info(f"📹 답변 영상 업로드: {sessionId}, Q{questionNumber}")
        
        # 1. GCS 업로드
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # 파일 확장자 확인 (기본값: webm)
        file_extension = ".webm"
        content_type = "video/webm"
        
        if videoFile.filename:
            if videoFile.filename.endswith('.mp4'):
                file_extension = ".mp4"
                content_type = "video/mp4"
            elif videoFile.filename.endswith('.webm'):
                file_extension = ".webm"
                content_type = "video/webm"
        
        video_filename = f"{sessionId}_q{questionNumber}{file_extension}"
        video_path = f"video/{video_filename}"
        
        blob = bucket.blob(video_path)
        video_content = await videoFile.read()
        blob.upload_from_string(video_content, content_type=content_type)
        
        video_uri = f"gs://{BUCKET_NAME}/{video_path}"
        logger.info(f"✅ 영상 업로드: {video_uri} ({content_type})")
        
        # 2. Progress JSON 업데이트
        update_progress_video_url(sessionId, questionNumber, video_uri)
        
        # 3. STT (영상 → 텍스트)
        answer_text = await video_to_text(video_uri)
        logger.info(f"✅ STT 완료: {answer_text[:100]}...")
        
        # 4. 세션 Agent에 텍스트 전달
        # ✅ 모든 호출에 [SESSION_ID: xxx] 포함 (파일명 매칭용)
        # ✅ is_first_call=False → 저장된 ADK 세션 재사용
        message_with_session = f"{answer_text}\n[SESSION_ID: {sessionId}]"
        response = await call_session_agent(message_with_session, session_id=sessionId, is_first_call=False)
        
        # 5. 응답 처리
        status = response.get("status")
        
        if status == "completed":
            logger.info(f"🎉 면접 종료: {sessionId}")
            return InterviewAnswerResponse(
                status="completed",
                sessionId=sessionId,
                remainingSlots=0,
                message=response.get("message", "면접이 종료되었습니다."),
                audioData=None  # 종료 시에는 오디오 없음
            )
        else:
            logger.info(f"➡️ 다음 질문: Q{response.get('questionId')}")
            
            # ✅ TTS: 다음 질문을 음성으로 변환 (면접관 스타일)
            question_text = response.get("question", "")
            is_tail = response.get("isTailQuestion", False)
            audio_data = None
            if question_text:
                audio_data = text_to_speech(question_text, is_tail_question=is_tail)
            
            return InterviewAnswerResponse(
                status="continue",
                questionId=response.get("questionId"),
                question=question_text,
                isTailQuestion=is_tail,
                sessionId=sessionId,
                remainingSlots=response.get("remainingSlots", 0),
                audioData=audio_data  # ✅ Base64 인코딩된 오디오 (면접관 스타일)
            )
        
    except Exception as e:
        logger.error(f"❌ 답변 처리 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"답변 처리 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/status/{session_id}")
async def get_interview_status(session_id: str):
    """
    면접 진행 상태 조회
    
    **Returns:**
    - sessionId: 세션 ID
    - totalQuestions: 총 질문 수
    - currentQuestionNumber: 현재 질문 번호
    - remainingSlots: 남은 질문 개수
    - askedQuestions: 물어본 질문 수
    - questions: 질문 리스트
    """
    
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        progress_path = f"progress_interview/{session_id}_progress.json"
        blob = bucket.blob(progress_path)
        
        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail="세션을 찾을 수 없습니다."
            )
        
        progress_data = json.loads(blob.download_as_text())
        
        return progress_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 상태 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"상태 조회 중 오류가 발생했습니다: {str(e)}"
        )
