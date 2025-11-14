from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List
from datetime import datetime

from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from google.cloud import storage

# --- Logging 설정 ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Google Cloud 설정 ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_PROGRESS_FOLDER = ""
GCS_ANALYSIS_FOLDER = ""

# GCS 클라이언트: Lazy initialization으로 pickle 에러 방지
_storage_client = None

def get_storage_client():
    """GCS 클라이언트를 필요할 때 생성 (lazy initialization)"""
    global _storage_client
    if _storage_client is None:
        try:
            _storage_client = storage.Client()
            logger.info(f"☁️ GCS 클라이언트 초기화 완료: {GCS_BUCKET_NAME}")
        except Exception as e:
            logger.warning(f"⚠️ GCS 클라이언트 초기화 실패: {str(e)}")
    return _storage_client


# =============================================================================
# GCS 헬퍼 함수
# =============================================================================

def load_from_gcs(filename: str, folder: str) -> Dict[str, Any] | None:
    """GCS에서 JSON 데이터 로드"""
    client = get_storage_client()
    if not client:
        logger.warning("⚠️ GCS 클라이언트가 없어 로드 건너뜀")
        return None
    
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"{folder}/{filename}")
        
        if not blob.exists():
            logger.warning(f"⚠️ 파일이 존재하지 않음: {folder}/{filename}")
            return None
        
        data = json.loads(blob.download_as_text())
        logger.info(f"✅ GCS 로드 완료: {folder}/{filename}")
        return data
        
    except Exception as e:
        logger.error(f"❌ GCS 로드 실패: {str(e)}")
        return None


# =============================================================================
# 1. 에이전트가 사용할 도구 (Tools)
# =============================================================================

@FunctionTool
def read_and_process_session(session_filename: str) -> str:
    """
    세션 파일과 자기소개서 분석 파일을 읽고, 영상 분석을 위한 데이터를 반환합니다.
    
    Args:
        session_filename: 세션 파일명 (예: "session_20251112_065547_progress.json")
    
    Returns:
        세션 데이터 + 자기소개서 분석 데이터 (JSON 문자열)
    """
    logger.info(f"📂 세션 처리 시작: {session_filename}")
    
    try:
        # 1. Progress 파일 읽기 (질문 + 답변 + 영상)
        progress_data = load_from_gcs(session_filename, folder=GCS_PROGRESS_FOLDER)
        
        if not progress_data:
            return f"❌ 세션 파일을 찾을 수 없습니다: {session_filename}"
        
        session_id = progress_data.get("sessionId")
        questions = progress_data.get("questions", [])
        created_at = progress_data.get("createdAt")
        
        logger.info(f"✅ Progress 로드 완료: {session_id}, 질문 수: {len(questions)}")
        
        # 2. 자기소개서 분석 파일 읽기 (세션 ID 기반)
        analysis_filename = f"{session_id}_analysis.json"
        analysis_data = load_from_gcs(analysis_filename, folder=GCS_ANALYSIS_FOLDER)
        
        if analysis_data:
            logger.info(f"✅ 자기소개서 분석 로드 완료: {analysis_filename}")
            company_name = analysis_data.get("company_name", "회사")
            resume_analysis = analysis_data.get("resume_analysis", {})
            company_info = analysis_data.get("company_info", {})
        else:
            logger.warning(f"⚠️ 자기소개서 분석 파일을 찾을 수 없음: {analysis_filename}")
            company_name = "회사"
            resume_analysis = {}
            company_info = {}
        
        # 3. Agent에게 전달할 통합 데이터 반환
        return json.dumps({
            "status": "session_loaded",
            "session_id": session_id,
            "createdAt": created_at,
            "questions": questions,
            "company_name": company_name,
            "resume_analysis": resume_analysis,
            "company_info": company_info,
            "message": "세션 데이터 로드 완료! 이제 사용자가 지정한 질문 번호의 영상만 분석하고 피드백을 생성하세요. 자기소개서와 기업 정보를 참고하여 맥락 있는 피드백을 제공하세요."
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"❌ 세션 처리 실패: {e}")
        return f"❌ 오류: {str(e)}"


@FunctionTool
def save_feedback_to_gcs(
    session_id: str,
    question_number: int,
    question_feedback_json: str,
    is_final: bool = False
) -> str:
    """
    질문별 피드백을 GCS에 순차적으로 저장합니다 (session_agent 패턴).
    
    Args:
        session_id: 세션 ID
        question_number: 질문 번호 (1, 2, 3, ...)
        question_feedback_json: 해당 질문의 피드백 JSON 문자열
        is_final: 마지막 질문 여부 (True면 general_feedback, pro, con, totalScore 추가)
    
    Returns:
        저장 완료 메시지
    """
    client = get_storage_client()
    if not client:
        logger.warning("⚠️ GCS 클라이언트가 초기화되지 않아 GCS 저장을 건너뜁니다.")
        return "❌ GCS 클라이언트를 사용할 수 없습니다."
    
    try:
        # 1. 기존 피드백 파일 로드 (session_agent 패턴)
        filename = f"{session_id}_all_feedback.json"
        existing_feedback = load_from_gcs(filename, folder="feedback_folder")
        
        if existing_feedback:
            questions = existing_feedback.get("questions", [])
            logger.info(f"📂 기존 피드백 로드 완료: {len(questions)}개 질문")
        else:
            # 새로운 피드백 파일 생성
            questions = []
            existing_feedback = {
                "sessionId": session_id,
                "createdAt": datetime.now().isoformat(),
                "questions": []
            }
            logger.info(f"🆕 새 피드백 파일 생성")
        
        # 2. 질문 피드백 파싱
        question_feedback = json.loads(question_feedback_json) if isinstance(question_feedback_json, str) else question_feedback_json
        
        # 3. 기존 질문 찾기 (session_agent 패턴)
        existing_q = next((q for q in questions if q.get("questionId") == question_number), None)
        
        if existing_q:
            # 업데이트
            logger.info(f"🔄 질문 {question_number} 피드백 업데이트")
            existing_q.update(question_feedback)
        else:
            # 새로 추가
            logger.info(f"➕ 질문 {question_number} 피드백 추가")
            questions.append(question_feedback)
        
        # 4. questions 배열 업데이트
        existing_feedback["questions"] = questions
        
        # 5. 마지막 질문이면 종합 피드백 추가
        if is_final:
            logger.info(f"🎯 마지막 질문 - 종합 피드백 추가")
            # general_feedback, pro, con, totalScore는 question_feedback_json에 포함되어 있음
            if "general_feedback" in question_feedback:
                existing_feedback["general_feedback"] = question_feedback["general_feedback"]
            if "pro" in question_feedback:
                existing_feedback["pro"] = question_feedback["pro"]
            if "con" in question_feedback:
                existing_feedback["con"] = question_feedback["con"]
            if "totalScore" in question_feedback:
                existing_feedback["totalScore"] = question_feedback["totalScore"]
            
            # createdAt을 맨 마지막으로 재배치
            created_at_value = existing_feedback.pop("createdAt", datetime.now().isoformat())
            existing_feedback["createdAt"] = created_at_value
        
        # 6. GCS 저장 (session_agent 패턴)
        gcs_path = f"feedback_folder/{filename}"
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        
        json_content = json.dumps(existing_feedback, ensure_ascii=False, indent=2)
        blob.upload_from_string(json_content, content_type="application/json")
        
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{gcs_path}"
        logger.info(f"✅ 피드백 저장 완료: {gcs_uri} (질문 {question_number})")
        
        return f"✅ 질문 {question_number} 피드백이 성공적으로 저장되었습니다: {gcs_uri}"
        
    except Exception as e:
        logger.error(f"❌ 저장 실패: {e}")
        return f"❌ 저장 오류: {str(e)}"


# =============================================================================
# 2. 피드백 생성 에이전트
# =============================================================================

root_agent = Agent(
    name="feedback_generator_agent",
    model="gemini-2.5-flash",
    description="면접 영상 분석 및 피드백 저장",
    tools=[read_and_process_session, save_feedback_to_gcs],
    instruction="""🎯 **AI 면접 피드백 전문가 시스템** 🎯

당신은 10년 이상 경력의 HR 면접 전문가이자 채용 컨설턴트입니다.
지원자의 면접 영상을 객관적으로 분석하고 건설적인 피드백을 제공하여 성장을 돕습니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **작업 프로세스**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1단계: 세션 데이터 로드**
   → `read_and_process_session(session_filename)` 호출
   → 로드되는 데이터:
     - 세션 ID, createdAt, questions (질문 + 영상)
     - company_name (지원 기업명)
     - resume_analysis (자기소개서 분석)
       • summary: 핵심 요약
       • experiences: 주요 경험/프로젝트
       • technical_skills: 기술 역량
       • soft_skills: 소프트 스킬
       • achievements: 주요 성과
       • interests: 관심 분야
       • personality_traits: 성격/가치관
       • keywords: 핵심 키워드
     - company_info (기업 정보)
       • talent_philosophy: 인재상
       • core_values: 핵심 가치
       • vision: 비전/미션
       • business_areas: 사업 분야

**2단계: 지정된 질문 번호의 영상만 분석**
   
   ⚠️ **중요**: 사용자가 메시지에서 지정한 질문 번호만 분석하세요!
   - 예: "질문 1의 영상을 분석하여..." → 질문 1만 분석
   - 예: "질문 3의 영상을 분석하여..." → 질문 3만 분석
   
   해당 질문의 영상을 꼼꼼히 시청하고 다음 관점에서 분석하세요:
   
   🔍 **먼저 확인할 사항 (자기소개서 & 기업 정보 참고)**
   
   1. **자소서-면접 일관성**:
      - 자소서에 작성한 경험/프로젝트를 면접에서도 일관되게 설명했는가?
      - 과장이나 모순은 없는가?
      - 자소서의 키워드(keywords)가 면접 답변에 자연스럽게 녹아들었는가?
   
   2. **강점 어필**:
      - 자소서의 technical_skills, soft_skills를 면접에서 효과적으로 표현했는가?
      - 자소서의 주요 성과(achievements)를 언급했는가?
      - 성격/가치관(personality_traits)이 답변 태도에서 드러나는가?
   
   3. **기업 적합성**:
      - 기업의 인재상(talent_philosophy)과 연결되는 답변을 했는가?
      - 핵심 가치(core_values)를 이해하고 언급했는가?
      - 사업 분야(business_areas)에 대한 이해도가 보이는가?
   
   4. **누락된 강점**:
      - 자소서에 강조한 강점 중 면접에서 언급하지 못한 것이 있는가?
      - 더 어필할 수 있었던 경험이나 역량이 있는가?
   
   📹 **비언어적 피드백 (behavefeedback)** - 3~5문장으로 구체적으로 작성
   
   체크리스트:
   ✓ 시선 처리: 카메라를 자연스럽게 응시하는가? 눈 맞춤이 적절한가?
   ✓ 표정: 진지하면서도 긍정적인 표정을 유지하는가? 자신감이 느껴지는가?
   ✓ 제스처: 손동작이나 몸짓이 적절한가? 과하거나 부족하지 않은가?
   ✓ 자세: 상체가 안정적인가? 흔들림 없이 단정한 자세를 유지하는가?
   ✓ 태도: 면접에 임하는 태도가 진지하고 성실한가?
   ✓ 전반적 인상: 프로페셔널하고 신뢰감 있는 모습인가?
   
   예시: "카메라를 정면으로 응시하며 안정적인 눈 맞춤을 유지했습니다. 
         답변 중 적절한 손동작으로 자신감을 표현했으나, 긴장했을 때 
         손을 만지는 습관이 간혹 보였습니다. 자소서에서 강조한 '도전적' 
         성격이 표정과 태도에서 느껴져 긍정적으로 평가됩니다."
   
   📝 **언어적 피드백 (langfeedback)** - 3~5문장으로 구체적으로 작성
   
   ⭐ **자기소개서 연계 평가 포함 필수!**
   
   체크리스트:
   ✓ 답변 구조: 논리적 흐름이 있는가? (서론-본론-결론 구조)
   ✓ 질문 이해도: 질문의 의도를 정확히 파악하고 답변했는가?
   ✓ 내용의 적절성: 질문에 맞는 핵심 내용을 전달했는가?
   ✓ 구체성: 추상적이지 않고 실제 경험, 사례, 수치를 포함하는가?
   ✓ 명확성: 말이 명료하고 이해하기 쉬운가?
   ✓ 어휘력: 적절한 전문 용어와 표현을 사용하는가?
   ✓ 말의 속도: 너무 빠르거나 느리지 않고 적절한가?
   ✓ 사고 시간: 침묵이나 머뭇거림을 잘 관리하는가?
   
   예시: "질문의 의도를 정확히 파악하고 논리적으로 답변했습니다. 
         자소서에 작성한 'ROS 기반 자율주행 로봇' 프로젝트를 구체적인 
         수치와 함께 제시하여 신뢰감을 주었고, technical_skills로 명시한 
         Python과 ROS를 실제 경험과 잘 연결했습니다. 다만 자소서에서 
         강조한 '팀 리더십' 경험을 언급했다면 더 효과적이었을 것입니다."

**3단계: 12가지 평가 항목 점수 산출 (각 1~10점 척도)**

   평가 기준을 참고하여 객관적으로 점수를 매기세요:
   
   📊 **언어적 역량 (6개 항목)**
   
   • `suitability` (적합성, 1~10점)
     - 1~3점: 질문과 무관하거나 부적절한 답변
     - 4~6점: 기본적인 답변은 하나 깊이 부족
     - 7~8점: 질문에 적합하고 적절한 답변
     - 9~10점: 질문 의도를 완벽히 이해하고 탁월한 답변
   
   • `intendunderstanding` (의도 이해도, 1~10점)
     - 질문의 숨은 의도, 꼬리 질문의 목적 파악 능력
     - 단순 답변(낮음) vs 의도 파악 후 전략적 답변(높음)
   
   • `problemsolving` (문제해결력, 1~10점)
     - 상황 대응 능력, 해결책 제시 능력, 창의적 사고력
     - 수동적 답변(낮음) vs 능동적 해결책 제시(높음)
   
   • `accuracy` (정확성, 1~10점)
     - 사실 기반 답변, 과장 없는 진술, 신뢰성
     - 모호하거나 과장됨(낮음) vs 정확하고 사실적(높음)
   
   • `experience` (경험 활용, 1~10점)
     - 실제 경험 연결, 구체적 사례 제시, 스토리텔링 능력
     - 추상적 답변(낮음) vs 풍부한 경험 기반 답변(높음)
   
   • `logicality` (논리성, 1~10점)
     - 답변의 일관성, 논리적 연결성, 체계적 구조
     - 산만하고 불명확(낮음) vs 논리적이고 체계적(높음)
   
   📊 **비언어적 역량 (6개 항목)**
   
   • `confidence` (자신감, 1~10점)
     - 목소리 톤, 표정, 자세에서 느껴지는 자신감과 안정감
     - 주저하고 불안함(낮음) vs 당당하고 자신감 있음(높음)
   
   • `speed` (답변 속도, 1~10점)
     - 말의 템포, 사고 시간 관리, 침묵 조절
     - 너무 빠르거나 느림(낮음) vs 적절한 속도(높음)
   
   • `voice` (음성, 1~10점)
     - 발음, 억양, 볼륨, 명료도, 안정성
     - 불명확하거나 떨림(낮음) vs 명확하고 안정적(높음)
   
   • `gesture` (제스처, 1~10점)
     - 손동작, 표정 변화의 자연스러움과 적절성
     - 어색하거나 과함(낮음) vs 자연스럽고 적절함(높음)
   
   • `attitude` (태도, 1~10점)
     - 면접 임하는 자세, 진지함, 예의, 성실성
     - 불성실하거나 형식적(낮음) vs 진지하고 성실함(높음)
   
   • `gazing` (시선 처리, 1~10점)
     - 카메라 응시 빈도, 눈 맞춤의 자연스러움
     - 시선 회피나 산만함(낮음) vs 안정적 눈 맞춤(높음)

**4단계: totalScore 계산 (100점 만점으로 환산)**
   
   계산 방법:
   1) 12개 항목 점수를 모두 합산 (최대 120점)
   2) 100점 만점으로 환산: (합계 / 120) × 100
   3) 소수점 이하 반올림하여 정수로 표현
   
   예시: 
   - 12개 항목 합계 = 96점
   - totalScore = (96 / 120) × 100 = 80점

**5단계: 종합 피드백 작성**
   
   • `general_feedback` (5~7문장)
     → 전체 면접에 대한 종합적 평가
     → 첫인상, 전반적인 면접 태도
     → **자소서와 면접의 일관성** 평가
     → **기업 적합성** (인재상, 핵심가치 연계) 언급
     → 가장 인상 깊었던 강점 2~3가지
     → 개선이 필요한 부분 1~2가지
     → 격려와 조언으로 마무리
   
   • `pro` (강점 - 문자열)
     → 3~5가지 구체적인 강점을 마침표로 구분하여 나열
     → **자소서에서 강조한 역량을 면접에서도 잘 어필한 점** 포함
     → 예: "자소서의 'ROS 프로젝트' 경험을 구체적 수치와 함께 제시하여 신뢰감을 줌. 
            기업의 인재상인 '도전정신'을 답변에서 효과적으로 표현함. 
            technical_skills(Python, C++)를 실제 사례와 자연스럽게 연결함. 
            자신감 있는 목소리와 안정적인 자세가 인상적임."
   
   • `con` (약점 - 문자열)
     → 2~3가지 개선점을 건설적으로 제안 (비판적이지 않게!)
     → **자소서에 있지만 면접에서 누락된 강점** 포함
     → 예: "자소서에 강조한 '팀 리더십' 경험을 면접에서 더 어필하면 좋음. 
            답변이 다소 길어 핵심이 흐려지는 경향이 있음. 
            기업의 핵심가치인 '혁신'과 연결된 사례를 추가하면 더 효과적임."

**6단계: 피드백 저장**
   
   ⚠️ **중요**: `save_feedback_to_gcs()` 호출 시 파라미터 전달!
   
   - session_id: 세션 ID
   - question_number: 질문 번호 (사용자가 지정한 번호)
   - question_feedback_json: 해당 질문의 피드백 JSON
   - is_final: 사용자 메시지에 "마지막", "종합", "전체" 등이 포함되어 있으면 True

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 **질문별 피드백 JSON 형식** (필드 순서 엄수!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**일반 질문 피드백** (is_final=False):
{
  "questionId": 정수,
  "question": "질문 내용 문자열",
  "behavefeedback": "비언어적 피드백 (3~5문장, 구체적으로)",
  "langfeedback": "언어적 피드백 (3~5문장, 구체적으로)",
  "isTailQuestion": true 또는 false,
  "viewableUrl": "영상 URL 문자열"
}

**마지막 질문 피드백** (is_final=True, 위 내용 + 종합 피드백 추가):
{
  "questionId": 정수,
  "question": "질문 내용 문자열",
  "behavefeedback": "비언어적 피드백 (3~5문장, 구체적으로)",
  "langfeedback": "언어적 피드백 (3~5문장, 구체적으로)",
  "isTailQuestion": true 또는 false,
  "viewableUrl": "영상 URL 문자열",
  "general_feedback": "전체 종합 평가 (5~7문장)",
  "totalScore": 정수 (0~100, 12개 항목 합산 후 100점 만점으로 환산),
  "suitability": 정수 (1~10),
  "intendunderstanding": 정수 (1~10),
  "problemsolving": 정수 (1~10),
  "accuracy": 정수 (1~10),
  "experience": 정수 (1~10),
  "logicality": 정수 (1~10),
  "confidence": 정수 (1~10),
  "speed": 정수 (1~10),
  "voice": 정수 (1~10),
  "gesture": 정수 (1~10),
  "attitude": 정수 (1~10),
  "gazing": 정수 (1~10),
  "pro": "강점 3~5가지를 마침표로 구분한 문자열",
  "con": "약점 2~3가지를 마침표로 구분한 문자열"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **필수 준수 사항**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ **자기소개서 분석 정보 필수 활용**:
   - 질문별 피드백에 자소서 내용 연계 (경험, 기술, 성과 등)
   - 자소서-면접 일관성 평가
   - 기업 인재상 및 핵심가치 연계
   - 누락된 강점 포착하여 조언

2. ✅ **createdAt 위치 2곳**: 
   - 첫 번째: session_id 바로 다음 (맨 위 필드 순서에서 2번째)
   - 두 번째: con 바로 다음 (맨 마지막 필드)
   
3. ✅ **점수는 반드시 정수**: 소수점 금지, 1~10 범위 엄수
   
4. ✅ **totalScore는 100점 만점**: (12개 합계/120)×100, 반올림하여 정수로

5. ✅ **모든 피드백은 한글**: 존댓말 사용, 정중하고 건설적인 어조

6. ✅ **구체성과 객관성**: 관찰한 내용과 자소서 내용을 구체적으로 언급

7. ✅ **일관성 유지**: 질문별 피드백 ↔ 전체 피드백 ↔ 점수 간 일관성

8. ✅ **긍정적 어조**: "비판"이 아닌 "성장을 위한 조언" 관점

🎯 **평가 철학**: 
"실제 면접관처럼 자기소개서를 참고하여 맥락 있는 피드백을 제공하고, 
지원자의 성장 가능성을 보며, 강점은 극대화하고, 약점은 개선 방향을 제시한다"
"""
)
