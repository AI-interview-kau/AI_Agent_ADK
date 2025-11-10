"""
Phase 1: 질문 생성 라우터
자기소개서 업로드 → Agent 호출 → 질문 생성
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

import vertexai
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from google.cloud import storage

logger = logging.getLogger(__name__)

# 환경 설정 - 환경변수에서 로드 (기본값 없음)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
QUESTION_AGENT_ID = os.getenv("QUESTION_AGENT_ID")

if not all([PROJECT_ID, BUCKET_NAME, QUESTION_AGENT_ID]):
    raise ValueError(
        "환경변수를 .env 파일에 설정해주세요: "
        "GOOGLE_CLOUD_PROJECT, GCS_BUCKET_NAME, QUESTION_AGENT_ID"
    )

router = APIRouter(prefix="/api", tags=["질문 생성"])


# ========== Response 모델 ==========

class GenerateQuestionsResponse(BaseModel):
    """질문 생성 응답"""
    status: str
    message: str
    sessionId: Optional[str] = None  # ✅ 세션 ID 추가
    question_count: Optional[int] = None
    gcs_uri: Optional[str] = None
    company_name: Optional[str] = None
    pdf_path: Optional[str] = None
    timestamp: Optional[str] = None


# ========== 엔드포인트 ==========

@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(resume_file: UploadFile = File(...)):
    """
    자기소개서 PDF 업로드 및 면접 질문 생성
    
    **Process:**
    1. PDF 파일 검증
    2. GCS 버킷에 저장 (pdf/)
    3. Pre-signed URL 생성
    4. 배포된 질문 생성 Agent 호출
    5. 질문 생성 대기 (비동기)
    6. 결과 반환
    
    **Returns:**
    - status: "success" | "error"
    - message: 결과 메시지
    - question_count: 생성된 질문 개수
    - company_name: 지원 기업명
    - gcs_uri: GCS 저장 경로
    """
    
    try:
        logger.info("=" * 60)
        logger.info("📤 질문 생성 요청 수신")
        
        # 1. PDF 검증
        if not resume_file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="PDF 파일만 업로드 가능합니다."
            )
        
        logger.info(f"📄 파일명: {resume_file.filename}")
        
        # ✅ 세션 ID 생성 (최우선!)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp_str}"
        logger.info(f"🆔 세션 ID 생성: {session_id}")
        
        # 2. GCS에 PDF 저장
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        pdf_filename = f"{session_id}_resume.pdf"  # ← 세션 ID 포함
        pdf_path = f"pdf/{pdf_filename}"
        
        logger.info(f"⬆️ GCS 업로드: {pdf_path}")
        
        blob = bucket.blob(pdf_path)
        pdf_content = await resume_file.read()
        blob.upload_from_string(pdf_content, content_type="application/pdf")
        
        logger.info(f"✅ 업로드 완료 ({len(pdf_content):,} bytes)")
        
        # 3. GCS URI 생성 (Presigned URL 대신 gs:// 직접 사용)
        gcs_uri = f"gs://{BUCKET_NAME}/{pdf_path}"
        logger.info(f"📦 GCS URI 생성: {gcs_uri}")
        
        # 4. Agent 호출
        logger.info(f"🤖 질문 생성 Agent 호출...")
        
        client = vertexai.Client(project=PROJECT_ID, location=LOCATION)
        # 프로젝트 ID를 사용 (프로젝트 넘버는 자동으로 해석됨)
        agent_resource_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{QUESTION_AGENT_ID}"
        adk_app = client.agent_engines.get(name=agent_resource_name)
        
        # ✅ 세션 ID 포함 메시지 (gs:// URI 전달)
        user_message = f"""안녕하세요! 자기소개서 PDF를 업로드했습니다.

[SESSION_ID: {session_id}]

GCS URI: {gcs_uri}

이 자기소개서를 분석하고, 지원 기업을 파악한 후, 
해당 기업 정보를 웹 검색하여 면접 분석 데이터를 GCS에 저장해주세요.

자동으로 모든 단계를 진행하고 GCS에 저장까지 완료해주세요!"""
        
        # 5. Agent 실행
        events = []
        async for event in adk_app.async_stream_query(
            user_id="web_user",
            message=user_message
        ):
            events.append(event)
        
        logger.info(f"✅ Agent 실행 완료 ({len(events)}개 이벤트)")
        
        # 6. Agent 응답 확인
        if events:
            last_event = events[-1]
            logger.info(f"📦 마지막 이벤트: {type(last_event).__name__}")
            if hasattr(last_event, 'content'):
                logger.info(f"📝 Agent 응답 미리보기: {str(last_event)[:500]}...")
        
        # 7. GCS 저장 대기 (10초로 증가)
        logger.info("⏳ GCS 저장 대기 중 (10초)...")
        await asyncio.sleep(10)
        
        # 8. ✅ 생성된 분석 파일 확인 (세션 ID 기반)
        analysis_filename = f"{session_id}_analysis.json"
        analysis_path = f"interview_questions/{analysis_filename}"
        
        logger.info(f"🔍 분석 파일 검색: {analysis_path}")
        
        analysis_blob = bucket.blob(analysis_path)
        
        if not analysis_blob.exists():
            error_detail = (
                f"분석 파일이 생성되지 않았습니다: {analysis_filename}. "
                f"Agent 이벤트: {len(events)}개. "
                f"Agent가 save_resume_analysis 및 update_company_info 함수를 호출했는지 확인하세요."
            )
            logger.error(f"❌ {error_detail}")
            
            # Agent 이벤트 상세 출력
            for i, event in enumerate(events[-3:], start=len(events)-2):
                logger.error(f"  Event [{i}]: {str(event)[:200]}")
            
            raise HTTPException(
                status_code=500,
                detail=error_detail
            )
        
        logger.info(f"📂 분석 파일 확인: {analysis_blob.name} (크기: {analysis_blob.size} bytes)")
        
        # 파일 내용 로드
        analysis_text = analysis_blob.download_as_text()
        logger.info(f"📄 파일 내용 길이: {len(analysis_text)} chars")
        
        if not analysis_text.strip():
            raise HTTPException(
                status_code=500,
                detail="분석 파일이 비어있습니다. Agent 로그를 확인하세요."
            )
        
        analysis_data = json.loads(analysis_text)
        
        company_name = analysis_data.get("company_name", "Unknown")
        # ✅ 분석 파일이므로 question_count는 없음 (나중에 session_agent에서 질문 생성)
        
        logger.info(f"✅ 자기소개서 및 기업 분석 완료: {company_name}")
        logger.info(f"   세션 ID: {session_id}")
        logger.info("=" * 60)
        
        return GenerateQuestionsResponse(
            status="success",
            message=f"자기소개서 및 기업 분석이 완료되었습니다.",
            sessionId=session_id,  # ✅ 세션 ID 추가
            question_count=None,  # 분석 단계에서는 질문 수 없음
            gcs_uri=f"gs://{BUCKET_NAME}/{analysis_blob.name}",
            company_name=company_name,
            pdf_path=pdf_path,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 에러: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"질문 생성 중 오류가 발생했습니다: {str(e)}"
        )
