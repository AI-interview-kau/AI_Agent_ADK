from __future__ import annotations

# .env 파일을 맨 먼저 로드합니다
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent
from google.cloud import storage

# --- Logging 설정 ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Google Cloud 설정 ---
# 환경변수에서 로드 (기본값 없음 - 반드시 .env에서 설정해야 함)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- GCS 설정 ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

if not PROJECT_ID or not GCS_BUCKET_NAME:
    raise ValueError("환경변수 GOOGLE_CLOUD_PROJECT와 GCS_BUCKET_NAME을 .env 파일에 설정해주세요.")

# =============================================================================
# 1. 에이전트가 사용할 도구 (Tools)
# =============================================================================

@FunctionTool
def get_individual_feedbacks(session_id: str) -> List[Dict[str, Any]]:
    """저장된 모든 개별 질문 피드백 JSON 파일들을 읽어와 리스트로 반환합니다."""
    
    logger.info(f"🔍 GCS에서 '{session_id}'의 개별 피드백 조회 중...")
    try:
        storage_client = storage.Client()
        # feedback_results/ 폴더에서 {session_id}_q*_feedback.json 패턴 검색
        prefix = f"feedback_results/{session_id}_q"
        blobs = storage_client.bucket(GCS_BUCKET_NAME).list_blobs(prefix=prefix)
        
        feedbacks = []
        for blob in blobs:
            if blob.name.endswith("_feedback.json") and "_final.json" not in blob.name:
                feedbacks.append(json.loads(blob.download_as_string()))
        
        if not feedbacks:
            raise RuntimeError(f"피드백 파일이 없습니다: gs://{GCS_BUCKET_NAME}/{prefix}")
        
        logger.info(f"✅ 총 {len(feedbacks)}개의 개별 피드백 로드 완료.")
        return feedbacks
    except Exception as e:
        logger.error(f"❌ GCS 피드백 조회 실패: {e}")
        raise RuntimeError(f"GCS 피드백 조회 실패: {e}")

@FunctionTool
def save_json_to_gcs(
    data: Dict[str, Any], 
    filename: str, 
    folder: str = "feedback_results"
) -> str:
    """JSON 데이터를 GCS 버킷에 저장합니다."""
    
    gcs_path = f"{folder}/{filename}"
    logger.info(f"⬆️ GCS에 '{gcs_path}' 저장 시도...")
    try:
        storage_client = storage.Client()
        blob = storage_client.bucket(GCS_BUCKET_NAME).blob(gcs_path)
        json_string = json.dumps(data, ensure_ascii=False, indent=2)
        blob.upload_from_string(json_string, content_type="application/json")
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{gcs_path}"
        logger.info(f"✅ GCS 저장 완료: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"❌ GCS 저장 실패: {e}")
        raise RuntimeError(f"GCS 저장 실패: {e}")


# =============================================================================
# 2. 전문 에이전트 (Sub-Agents) 정의
# =============================================================================

# --- 2-1. 개별 영상 분석 에이전트 ---
individual_feedback_agent = Agent(
    name="individual_feedback_agent",
    model="gemini-2.5-pro",
    description="하나의 답변 영상을 받아 언어적(STT)/행동적 피드백 생성",
    tools=[save_json_to_gcs],
    instruction="""
당신은 AI 면접 코치입니다. 당신의 임무는 **단 하나의** 답변 영상을 분석하는 것입니다.

**입력:**
- `session_id`: 세션 ID (예: "session123")
- `question_number`: 질문 번호 (예: 1)
- `video_url`: GCS 영상 경로 (예: "gs://bucket/videos/session123_q1.webm")
- `question`: 질문 내용 (예: "자기소개를 해주세요")

**중요:** Gemini 2.5 Pro는 GCS URL을 직접 분석할 수 있습니다!
- 다운로드, 파일 분리, 별도 도구 호출이 필요 없습니다.
- `video_url`을 직접 읽어서 오디오와 비디오를 동시에 분석하세요.

**작업 계획:**
1.  **[멀티모달 직접 분석]** `video_url` (GCS URL)에 있는 영상을 **직접** 분석합니다.
    - GCS URL을 그대로 사용하여 영상의 오디오와 비디오를 동시에 분석하세요.
    - 다운로드나 파일 변환 없이 바로 분석 가능합니다!

2.  **언어적 피드백 (langfeedback) 생성:**
    - 영상의 **오디오 채널**에서 음성을 텍스트로 변환(STT)합니다.
    - 답변의 논리성, 명확성, 구체성을 평가합니다.
    - 질문 내용(`question`)과 답변의 연관성을 평가합니다.
    - **3-5문장**으로 요약하여 피드백을 작성하세요.

3.  **행동적 피드백 (behaviorfeedback) 생성:**
    - 영상의 **비디오 채널**에서:
      * 시선 처리 (카메라를 보는지, 안정적인지)
      * 표정 (밝고 자연스러운지, 경직되어 있지 않은지)
      * 자세 (바르게 앉았는지, 움직임이 자연스러운지)
    - 영상의 **오디오 채널**에서:
      * 목소리 톤 (안정적이고 자신감 있는지)
      * 말의 빠르기 (너무 빠르거나 느리지 않은지)
      * 필러 사용 빈도 ("음", "어", "그" 등의 불필요한 단어)
    - **3-5문장**으로 요약하여 피드백을 작성하세요.

4.  **피드백 저장:** `save_json_to_gcs` 도구를 호출하여 생성한 피드백을 저장합니다:
    - `data`: 
      {
        "questionNumber": (입력받은 question_number),
        "sessionId": (입력받은 session_id),
        "question": (입력받은 question),
        "langfeedback": (2단계에서 생성한 언어적 피드백 텍스트),
        "behaviorfeedback": (3단계에서 생성한 행동적 피드백 텍스트)
      }
    - `folder`: "feedback_results"
    - `filename`: (입력받은 session_id) + "_q" + (입력받은 question_number) + "_feedback.json"
      예시: "session123_q1_feedback.json"

5.  **최종 결과 반환:** "개별 피드백 저장이 완료되었습니다."라고 응답합니다.

**주의사항:**
- 다운로드나 파일 변환 도구를 사용하지 마세요 (필요 없습니다!)
- GCS URL을 직접 분석하는 것이 가장 빠르고 효율적입니다.
- 피드백은 구체적이고 건설적이어야 합니다.
""",
)

# --- 2-2. 최종 피드백 생성 에이전트 ---
final_feedback_agent = Agent(
    name="final_feedback_agent",
    model="gemini-2.5-pro",
    description="모든 개별 피드백을 취합하여 최종 점수 리포트 생성",
    tools=[get_individual_feedbacks, save_json_to_gcs],
    instruction="""
당신은 AI 면접 평가 총괄 매니저입니다.
당신의 임무는 세션의 모든 개별 피드백을 취합하여 **최종 점수 리포트**를 생성하는 것입니다.

**입력:**
`session_id`를 받습니다.

**작업 계획:**
1.  `get_individual_feedbacks` 도구를 `session_id`로 호출하여 GCS에 저장된 모든 개별 피드백(`langfeedback`, `behaviorfeedback` 리스트)을 가져옵니다.

2.  [종합 분석] 모든 `langfeedback` (언어적 요약)과 `behaviorfeedback` (행동적 요약)을 종합적으로 검토합니다.

3.  [점수 산출] 각 피드백 내용을 바탕으로, 아래 **12개 항목**에 대해 **0점에서 100점 사이의 점수(정수)**를 산출합니다:
    
    **언어적 점수 (6개):**
    - `suitability`: 직무 적합성 점수 (0-100)
    - `intendunderstanding`: 질문 의도 파악 점수 (0-100)
    - `problemsolving`: 문제 해결 능력 점수 (0-100)
    - `accuracy`: 답변 정확성 점수 (0-100)
    - `experience`: 경험의 깊이 점수 (0-100)
    - `logicality`: 답변 논리성 점수 (0-100)
    
    **비언어적 점수 (6개):**
    - `confidence`: 자신감/표정 점수 (0-100)
    - `speed`: 말하기 속도 점수 (0-100)
    - `voice`: 목소리 톤/안정성 점수 (0-100)
    - `gesture`: 제스처/자세 점수 (0-100)
    - `attitude`: 태도/진정성 점수 (0-100)
    - `gazing`: 시선 처리 점수 (0-100)

4.  [최종 피드백 작성] 모든 개별 피드백을 종합하여 **전체 면접에 대한 상세한 피드백 텍스트**를 작성합니다.
    언어적 측면과 비언어적 측면을 모두 포함하여, 지원자의 강점과 개선점을 명확히 전달하세요.

5.  **최종 점수 리포트 (JSON) 생성:**
    {
      "sessionId": (입력받은 session_id),
      "createdAt": (현재 시간을 ISO 8601 형식으로, 예: "2025-11-06T10:30:00Z"),
      
      "suitability": (3단계에서 산출한 점수),
      "intendunderstanding": (3단계에서 산출한 점수),
      "problemsolving": (3단계에서 산출한 점수),
      "accuracy": (3단계에서 산출한 점수),
      "experience": (3단계에서 산출한 점수),
      "logicality": (3단계에서 산출한 점수),
      
      "confidence": (3단계에서 산출한 점수),
      "speed": (3단계에서 산출한 점수),
      "voice": (3단계에서 산출한 점수),
      "gesture": (3단계에서 산출한 점수),
      "attitude": (3단계에서 산출한 점수),
      "gazing": (3단계에서 산출한 점수),
      
      "general_feedback": (4단계에서 작성한 최종 피드백 텍스트)
    }

6.  **최종 피드백 저장:** `save_json_to_gcs` 도구를 호출하여 생성한 최종 리포트를 저장합니다:
    - `data`: (5단계에서 생성한 최종 점수 리포트 JSON)
    - `folder`: "feedback_results"
    - `filename`: (입력받은 session_id) + "_final.json"
      예: "session123_final.json"

7.  **최종 결과 반환:** "최종 피드백 저장이 완료되었습니다."라고 응답합니다.
""",
)

# =============================================================================
# 3. 루트 에이전트 (총괄 지휘자)
# =============================================================================

# 'adk web'이 실행할 수 있도록 이 파일을 대표하는 Root Agent로 정의합니다.
root_agent = Agent(
    name="video_feedback_orchestrator",
    model="gemini-2.5-pro",
    description="사용자의 요청을 분석하여 개별 영상 분석 또는 최종 피드백 생성을 지시하는 총괄 지휘자",
    tools=[transfer_to_agent],
    sub_agents=[individual_feedback_agent, final_feedback_agent],
    instruction="""
당신은 AI 면접 피드백 시스템의 총괄 지휘자입니다.
당신의 임무는 두 가지 요청을 구분하여 적절한 sub-agent에게 전달하는 것입니다.

**[흐름 1: 개별 영상 분석]**
- **입력 감지**: 사용자의 입력이 `{"session_id": ..., "question_number": ..., "video_url": ..., "question": ...}` 형식이면 이 흐름을 실행합니다.
- **작업 계획:**
    1.  `transfer_to_agent`를 호출하여 `individual_feedback_agent`에게 
        입력받은 JSON 객체 전체를 그대로 전달합니다.
    2.  individual_feedback_agent가 피드백을 분석하고 저장한 후 결과를 반환할 것입니다.
    3.  결과를 사용자에게 전달합니다.

**[흐름 2: 최종 피드백 생성]**
- **입력 감지**: 사용자의 입력이 "세션 종료" 신호이거나 `{"generate_final_feedback": true, "session_id": ...}` 또는 단순히 `{"session_id": ...}` 형식(question_number 없음)이면 이 흐름을 실행합니다.
- **작업 계획:**
    1.  `transfer_to_agent`를 호출하여 `final_feedback_agent`에게 입력 JSON을 그대로 전달합니다.
    2.  final_feedback_agent가 최종 리포트를 생성하고 저장한 후 결과를 반환할 것입니다.
    3.  결과를 사용자에게 전달합니다.

사용자의 입력 형식을 분석하여 위 2가지 흐름 중 하나를 정확히 실행하세요.
""",
)