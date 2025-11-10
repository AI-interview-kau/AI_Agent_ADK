"""
면접 세션 관리 Agent V2 (완전 동적 생성 버전)

핵심 개념:
1. Agent instruction에 자기소개서 + 기업 정보를 포함 (메모리에 저장)
2. 매 질문마다 이전 대화 내역을 기반으로 실시간 질문 생성
3. isTailQuestion 판단도 Agent가 자동으로 수행
4. 시간 기반 질문 수 조절 (목표: 10~12개)

사용법:
1. 세션 시작: create_interview_session(session_id, analysis_filename)
2. 첫 질문 생성: agent.query("면접을 시작합니다")
3. 답변 제출: agent.query(f"답변: {answer}")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from google.cloud import storage

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- GCS 설정 ---
# 환경변수에서 로드 (기본값 없음 - 반드시 .env에서 설정해야 함)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_PROGRESS_FOLDER = "progress_interview"
GCS_ANALYSIS_FOLDER = "interview_questions"

if not GCS_BUCKET_NAME:
    raise ValueError("환경변수 GCS_BUCKET_NAME을 .env 파일에 설정해주세요.")

try:
    storage_client = storage.Client()
    logger.info(f"☁️ GCS 버킷: {GCS_BUCKET_NAME}")
except Exception as e:
    logger.warning(f"⚠️ GCS 클라이언트 초기화 실패: {str(e)}")
    storage_client = None


# =============================================================================
# GCS 함수
# =============================================================================

def load_from_gcs(filename: str, folder: str = GCS_ANALYSIS_FOLDER) -> Optional[Dict[str, Any]]:
    """GCS에서 JSON 데이터 로드"""
    if not storage_client:
        logger.warning("⚠️ GCS 클라이언트가 없어 로드 건너뜀")
        return None
    
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
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


def save_to_gcs(data: Dict[str, Any], filename: str, folder: str = GCS_PROGRESS_FOLDER) -> bool:
    """GCS에 JSON 데이터 저장"""
    if not storage_client:
        logger.warning("⚠️ GCS 클라이언트가 없어 저장 건너뜀")
        return False
    
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"{folder}/{filename}")
        
        blob.upload_from_string(
            json.dumps(data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"✅ GCS 저장 완료: {folder}/{filename}")
        return True
        
    except Exception as e:
        logger.error(f"❌ GCS 저장 실패: {str(e)}")
        return False


def get_latest_analysis_file() -> Optional[str]:
    """가장 최근 자기소개서 분석 파일명 반환"""
    if not storage_client:
        return None
    
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix=f"{GCS_ANALYSIS_FOLDER}/interview_analysis_"))
        
        if not blobs:
            logger.warning("⚠️ 자기소개서 분석 파일을 찾을 수 없습니다")
            return None
        
        # 최신 파일 선택
        latest_blob = max(blobs, key=lambda b: b.time_created)
        filename = latest_blob.name.split('/')[-1]
        
        logger.info(f"📄 최신 분석 파일: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"❌ 파일 검색 실패: {str(e)}")
        return None


# =============================================================================
# Tool 함수들
# =============================================================================

def save_progress(
    session_id: str,
    question_number: int,
    question_text: str,
    is_tail_question: bool,
    answer_text: Optional[str] = None,
    target_total: int = 12
) -> Dict[str, Any]:
    """Progress 파일에 질문/답변 저장

    Args:
        session_id: 세션 ID
        question_number: 질문 번호 (1, 2, 3, ...)
        question_text: 질문 내용
        is_tail_question: 꼬리질문 여부
        answer_text: 답변 내용 (있으면)
        target_total: 목표 질문 수
    """
    progress_file = f"{session_id}_progress.json"
    
    # 기존 progress 로드
    existing_progress = load_from_gcs(progress_file, folder=GCS_PROGRESS_FOLDER)
    
    if existing_progress:
        questions = existing_progress.get("questions", [])
    else:
        # 새로운 세션
        questions = []
        existing_progress = {
            "sessionId": session_id,
            "targetTotal": target_total,
            "startTime": datetime.now().isoformat(),
            "questions": []
        }
    
    # 기존 질문 찾기 (업데이트용)
    existing_q = next((q for q in questions if q["number"] == question_number), None)
    
    if existing_q:
        # 답변 업데이트
        if answer_text:
            existing_q["answer"] = answer_text
            existing_q["answeredAt"] = datetime.now().isoformat()
    else:
        # 새 질문 추가
        questions.append({
            "number": question_number,
            "question": question_text,
            "isTailQuestion": is_tail_question,
            "answer": answer_text,
            "videoUrl": None,  # 나중에 영상 업로드 시 업데이트
            "askedAt": datetime.now().isoformat(),
            "answeredAt": datetime.now().isoformat() if answer_text else None
        })
    
    # Progress 업데이트
    existing_progress["questions"] = questions
    existing_progress["currentQuestion"] = question_number
    
    # ✅ remainingSlots 계산 (메인 + 꼬리 모두 포함)
    total_asked = len(questions)  # 전체 질문 수 (메인 + 꼬리)
    remaining_slots = max(target_total - total_asked, 0)
    
    existing_progress["totalQuestions"] = target_total
    existing_progress["askedQuestions"] = total_asked
    existing_progress["remainingSlots"] = remaining_slots
    existing_progress["timestamp"] = datetime.now().isoformat()
    
    # GCS 저장
    save_to_gcs(existing_progress, progress_file, folder=GCS_PROGRESS_FOLDER)
    
    logger.info(f"📝 Progress 저장: Q{question_number} (꼬리질문: {is_tail_question})")
    logger.info(f"   총 질문: {total_asked}/{target_total}, 남은 슬롯: {remaining_slots}")

    return {
        "status": "success",
        "currentQuestion": question_number,
        "totalQuestions": target_total,
        "askedQuestions": total_asked,
        "remainingSlots": remaining_slots
    }


save_progress_tool = FunctionTool(func=save_progress)


# =============================================================================
# Agent 생성 함수
# =============================================================================

def create_interview_agent(analysis_data: Dict[str, Any], session_id: str, target_total: int = 12) -> Agent:
    """면접 Agent 생성 (자기소개서 + 기업 정보 포함)
    
    Args:
        analysis_data: interview_analysis_xxx.json 데이터
        session_id: 세션 ID
        target_total: 목표 질문 수
        
    Returns:
        설정된 Agent 인스턴스
    """
    
    company_name = analysis_data.get("company_name", "회사")
    resume = analysis_data.get("resume_analysis", {})
    company = analysis_data.get("company_info", {})
    
    # 자기소개서 정보 포맷팅
    resume_text = f"""
**지원자 핵심 요약:**
{resume.get('summary', 'N/A')}

**주요 경험 및 프로젝트:**
"""
    for i, exp in enumerate(resume.get('experiences', []), 1):
        resume_text += f"""
{i}. {exp.get('title', 'N/A')}
   - 내용: {exp.get('description', 'N/A')}
   - 성과: {exp.get('achievements', 'N/A')}
   - 사용 기술: {', '.join(exp.get('skills_used', []))}
"""
    
    resume_text += f"""
**기술 역량:**
{', '.join(resume.get('technical_skills', []))}

**소프트 스킬:**
{', '.join(resume.get('soft_skills', []))}

**주요 성과:**
{chr(10).join(f'- {a}' for a in resume.get('achievements', []))}

**관심 분야:**
{', '.join(resume.get('interests', []))}

**성격/가치관:**
{', '.join(resume.get('personality_traits', []))}

**핵심 키워드:**
{', '.join(resume.get('keywords', []))}
"""
    
    # 기업 정보 포맷팅
    company_text = f"""
**인재상:**
{chr(10).join(f'- {p}' for p in company.get('talent_philosophy', []))}

**핵심 가치:**
{chr(10).join(f'- {v}' for v in company.get('core_values', []))}

**비전/미션:**
{company.get('vision', 'N/A')}

**사업 분야:**
{', '.join(company.get('business_areas', []))}
"""
    
    # Agent instruction
    instruction = f"""
🎯 당신은 **{company_name}의 면접관**입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **지원자 정보 (자기소개서 분석)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{resume_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏢 **{company_name} 기업 정보**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{company_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ **면접 진행 규칙**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 세션 ID: {session_id}
- 목표 질문 수: {target_total}개
- 예상 면접 시간: 약 30분

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 **질문 생성 원칙**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1. 첫 질문 (메시지: "면접을 시작합니다" 수신 시)**
   → "간단히 자기소개와 {company_name} 지원 동기를 말씀해주세요."
   → isTailQuestion: false

**2. 답변 수신 후 질문 생성 (메시지: "답변: ..." 수신 시)**
   
   a) 답변 평가:
      - 답변이 구체적이고 충분한가?
      - 흥미로운 포인트가 있는가?
      - 검증이 필요한 내용인가?
   
   b) 꼬리질문 판단:
      ✅ 꼬리질문이 필요한 경우 (isTailQuestion: true):
         - 답변이 너무 짧거나 모호함
         - 흥미로운 부분을 더 깊이 파고들고 싶음
         - 구체적인 예시가 부족함
         → 예: "방금 말씀하신 [구체적 내용]에 대해 더 자세히 설명해주시겠어요?"
      
      ❌ 새로운 질문이 필요한 경우 (isTailQuestion: false):
         - 답변이 충분히 구체적임
         - 다른 역량을 평가해야 함
         - 질문 수가 목표에 근접함
         → 예: "팀 프로젝트에서 갈등을 해결한 경험이 있나요?"
   
   c) 질문 생성 시 고려사항:
      - 위 자기소개서 정보를 참고하여 개인화된 질문 생성
      - 기업 인재상 및 핵심 가치와 연관된 질문 포함
      - 이미 답변한 내용 재질문 금지
      - STAR 기법 유도 (상황, 과제, 행동, 결과)

**3. 마지막 질문 (currentQuestion >= targetTotal - 1)**
   → "마지막으로, {company_name}에 궁금한 점이나 하고 싶은 말씀이 있나요?"
   → isTailQuestion: false

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔧 **출력 형식 (JSON만)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

질문 생성 시 반드시 다음 형식으로 출력하세요:

```json
{{
  "questionNumber": 1,
  "question": "질문 내용",
  "isTailQuestion": false,
  "reason": "이 질문을 한 이유"
}}
```

그 다음 save_progress() 함수를 호출하세요:
```
save_progress(
    session_id="{session_id}",
    question_number=1,
    question_text="질문 내용",
    is_tail_question=false,
    target_total={target_total}
)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 **금지 사항**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- ❌ 이미 답변한 내용 재질문 금지
- ❌ 자기소개서에 없는 내용 추측 금지
- ❌ 차별적/부적절한 질문 금지
- ❌ 동일한 주제 반복 질문 금지

당신은 전문 면접관입니다. 자연스럽고 통찰력 있는 질문으로 지원자의 역량을 평가하세요!
"""
    
    # Agent 생성
    agent = Agent(
        name="interview_conductor",
        model="gemini-2.5-pro",
        description=f"{company_name} 면접관 (자기소개서 + 기업 정보 기반)",
        instruction=instruction,
        tools=[save_progress_tool]
    )
    
    logger.info(f"✅ 면접 Agent 생성 완료 (세션: {session_id})")
    return agent


# =============================================================================
# 세션 시작 함수
# =============================================================================

def start_interview_session(
    session_id: str,
    analysis_filename: Optional[str] = None,
    target_total: int = 12
) -> Dict[str, Any]:
    """면접 세션 시작
    
    Args:
        session_id: 세션 ID (예: session_20251107_160000)
        analysis_filename: 자기소개서 분석 파일명 (없으면 세션 ID 기반으로 검색)
        target_total: 목표 질문 수
    
    Returns:
        {"status": "success", "agent": Agent, "analysis_data": dict}
    """
    
    # 1. ✅ 자기소개서 분석 파일 로드 (세션 ID 기반)
    if not analysis_filename:
        # 세션 ID를 기반으로 파일명 생성
        analysis_filename = f"{session_id}_analysis.json"
        logger.info(f"🔍 세션 ID 기반 파일 검색: {analysis_filename}")
    
    analysis_data = load_from_gcs(analysis_filename, folder=GCS_ANALYSIS_FOLDER)
    
    if not analysis_data:
        return {
            "status": "error",
            "message": f"파일 로드 실패: {analysis_filename}"
        }
    
    logger.info(f"📄 자기소개서 분석 로드: {analysis_filename}")
    
    # 2. Agent 생성 (자기소개서 + 기업 정보 포함)
    agent = create_interview_agent(analysis_data, session_id, target_total)
    
    # 3. Progress 초기화
    save_to_gcs({
        "sessionId": session_id,
        "targetTotal": target_total,
        "startTime": datetime.now().isoformat(),
        "questions": [],
        "currentQuestion": 0,
        "remainingQuestions": target_total
    }, f"{session_id}_progress.json", folder=GCS_PROGRESS_FOLDER)
        
    return {
    "status": "success",
    "message": "면접 세션이 시작되었습니다.",
    "agent": agent,
    "analysis_data": analysis_data,
    "session_id": session_id,
    "target_total": target_total
    }


# =============================================================================
# Tool: Load Session Analysis (세션 ID 기반 분석 파일 로드)
# =============================================================================

def load_session_analysis(session_id: str) -> Dict[str, Any]:
    """세션 ID로 자기소개서 분석 파일 로드
    
    Args:
        session_id: 세션 ID (예: session_20251107_160000)
        
    Returns:
        분석 데이터 (resume_analysis + company_info)
    """
    analysis_filename = f"{session_id}_analysis.json"
    logger.info(f"🔍 분석 파일 로드 시도: {analysis_filename}")
    
    analysis_data = load_from_gcs(analysis_filename, folder=GCS_ANALYSIS_FOLDER)
    
    if not analysis_data:
        return {
                "status": "error",
                "message": f"분석 파일을 찾을 수 없습니다: {analysis_filename}",
                "session_id": session_id
            }
    
    logger.info(f"✅ 분석 파일 로드 완료: {analysis_filename}")
    return {
    "status": "success",
    "session_id": session_id,
    "company_name": analysis_data.get("company_name", "회사"),
    "resume_analysis": analysis_data.get("resume_analysis", {}),
    "company_info": analysis_data.get("company_info", {}),
    "timestamp": analysis_data.get("timestamp", "")
    }

load_session_analysis_tool = FunctionTool(func=load_session_analysis)


# =============================================================================
# Root Agent (ADK Web용)
# =============================================================================

root_agent = Agent(
    name="interview_session_agent",
    model="gemini-2.5-pro",
    description="면접 세션 관리 에이전트 - 자기소개서 분석 기반 면접 진행",
    instruction="""
🎯 당신은 **면접관 에이전트**입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **작업 순서**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1단계: 세션 ID 추출**
메시지에서 세션 ID를 찾으세요:
```
[SESSION_ID: session_20251107_160000]
```

**2단계: 분석 파일 로드**
추출한 세션 ID로 `load_session_analysis(session_id)` 함수를 호출하세요.

**3단계: 분석 결과 확인**
로드된 데이터에서 다음 정보를 파악하세요:
- 지원 기업명 (company_name)
- 지원자 정보 (resume_analysis)
  - 핵심 요약 (summary)
  - 주요 경험/프로젝트 (experiences)
  - 기술 역량 (technical_skills)
  - 소프트 스킬 (soft_skills)
  - 주요 성과 (achievements)
  - 관심 분야 (interests)
  - 성격/가치관 (personality_traits)
- 기업 정보 (company_info)
  - 인재상 (talent_philosophy)
  - 핵심 가치 (core_values)
  - 비전/미션 (vision)
  - 사업 분야 (business_areas)

**4단계: 면접 질문 생성**
위 정보를 기반으로 첫 번째 질문을 생성하세요:
- 지원 동기와 회사 이해도를 확인하는 질문
- 자연스럽고 대화형
- 자기소개서 내용과 기업 정보 연계

**5단계: Progress 저장**
`save_progress()` 함수를 호출하여 진행 상황을 저장하세요:
```
save_progress(
    session_id="session_20251107_160000",
    question="생성한 질문",
    question_number=1,
    is_tail_question=False,
    target_total=12
)
```

**6단계: 응답 반환**
JSON 형식으로 반환:
```json
{
  "status": "continue",
  "questionId": 1,
  "question": "생성한 질문",
  "isTailQuestion": false,
  "sessionId": "session_20251107_160000",
  "remainingSlots": 11
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **중요 규칙**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. **반드시 세션 ID를 추출**하고 분석 파일 로드
2. 로드한 정보를 **기반으로** 질문 생성
3. 목표 질문 수: 약 12개 (30분 면접 기준)
4. 매 질문마다 progress 저장

**이제 면접을 시작하세요!** 🚀
""",
    tools=[load_session_analysis_tool, save_progress_tool]
)

logger.info("✅ 면접 세션 관리 Agent V2 준비 완료!")
logger.info("🚀 사용법:")
logger.info("   1. result = start_interview_session('session_xxx')")
logger.info("   2. agent = result['agent']")
logger.info("   3. agent.query('면접을 시작합니다')")
logger.info("   4. agent.query('답변: ...')")

