from __future__ import annotations
import requests
import base64
import io
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
from pydantic import BaseModel, Field

# --- Dependencies ---
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- GCS 설정 ---
# 환경변수에서 로드 (기본값 없음 - 반드시 .env에서 설정해야 함)
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

if not GCS_BUCKET_NAME or not PROJECT_ID:
    raise ValueError("환경변수 GCS_BUCKET_NAME과 GOOGLE_CLOUD_PROJECT를 .env 파일에 설정해주세요.")

try:
    # 배포 환경에서도 명시적으로 프로젝트 지정
    storage_client = storage.Client(project=PROJECT_ID)
    logger.info(f"☁️ GCS 버킷: {GCS_BUCKET_NAME}")
    logger.info(f"📌 프로젝트: {PROJECT_ID}")
except Exception as e:
    logger.warning(f"⚠️ GCS 클라이언트 초기화 실패: {str(e)}")
    storage_client = None


# =============================================================================
# GCS 저장 함수
# =============================================================================

def save_to_gcs(
    data: Dict[str, Any], 
    filename: str, 
    folder: str = "interview_questions"
) -> Optional[str]:
    """
    GCS에 JSON 데이터 저장
    
    Args:
        data: 저장할 데이터 (딕셔너리)
        filename: 파일명 (예: "interview_questions_user123_20251027.json")
        folder: GCS 내 폴더명 (기본값: "interview_questions")
    
    Returns:
        GCS URI (gs://interview-data-cosmic-mariner/...) 또는 None (실패 시)
    """
    logger.info(f"🔍 save_to_gcs 시작: {folder}/{filename}")
    
    if not storage_client:
        logger.warning("⚠️ GCS 클라이언트가 초기화되지 않아 GCS 저장을 건너뜁니다.")
        return None
    
    try:
        logger.info(f"📦 버킷 연결 중: {GCS_BUCKET_NAME}")
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        logger.info(f"📝 Blob 생성 중: {folder}/{filename}")
        blob = bucket.blob(f"{folder}/{filename}")
        
        # JSON 데이터를 문자열로 변환하여 업로드
        json_string = json.dumps(data, ensure_ascii=False, indent=2)
        logger.info(f"📊 JSON 크기: {len(json_string):,} bytes")
        
        logger.info(f"⬆️ GCS 업로드 시작...")
        blob.upload_from_string(
            json_string,
            content_type="application/json"
        )
        logger.info(f"⬆️ GCS 업로드 완료")
        
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{folder}/{filename}"
        logger.info(f"✅ GCS 저장 완료: {gcs_uri}")
        
        return gcs_uri
        
    except Exception as e:
        logger.error(f"❌ GCS 저장 실패: {str(e)}")
        logger.error(f"❌ 에러 타입: {type(e).__name__}")
        import traceback
        logger.error(f"❌ 상세 스택:\n{traceback.format_exc()}")
        return None


# =============================================================================
# TOOLS
# =============================================================================

# === TOOLS ===

# --- Tool 1: Resume Loading ---
class ResumeContentRequest(BaseModel):
    """자기소개서 로딩 요청"""
    pdf_base64: Optional[str] = Field(None, description="Base64 인코딩된 PDF")
    file_path: Optional[str] = Field(None, description="로컬 파일 경로")
    fallback_text: Optional[str] = Field(None, description="직접 입력한 텍스트")

class ResumeContent(BaseModel):
    """자기소개서 내용"""
    resume_text: str = Field(..., description="추출된 텍스트")
    page_count: int = Field(..., description="페이지 수")

def load_resume_content(
    pdf_base64: Optional[str] = None,
    file_path: Optional[str] = None, 
    fallback_text: Optional[str] = None
) -> Dict[str, Any]:
    """자기소개서 PDF 또는 텍스트를 로드합니다.
    
    Args:
        pdf_base64: Base64 인코딩된 PDF 데이터
        file_path: PDF 파일 경로 (gs://, http(s)://, 또는 로컬 경로)
                   - gs://bucket/path/file.pdf (GCS URI - 권장)
                   - http(s)://... (Presigned URL - 레거시)
                   - /local/path/file.pdf (로컬 파일)
        fallback_text: 직접 입력한 텍스트
        
    Returns:
        Dict containing resume_text and page_count
    """
    
    if pdf_base64:
        if not PdfReader:
            raise RuntimeError("pypdf가 설치되지 않았습니다.")
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                raise RuntimeError("PDF에서 텍스트를 추출할 수 없습니다.")
            return {"resume_text": text, "page_count": len(reader.pages)}
        except Exception as e:
            raise RuntimeError(f"PDF 처리 중 오류 발생: {str(e)}")
    # 2. GCS URI에서 PDF 다운로드 (gs://bucket/path 형식)
    if file_path and file_path.startswith("gs://"):
        if not PdfReader:
            raise RuntimeError("pypdf가 설치되지 않았습니다.")
        if not storage_client:
            raise RuntimeError("GCS 클라이언트가 초기화되지 않았습니다.")
        
        try:
            logger.info(f"📥 GCS에서 PDF 다운로드 중: {file_path}")
            
            # gs://bucket-name/path/to/file.pdf → bucket-name, path/to/file.pdf
            gcs_path = file_path.replace("gs://", "")
            bucket_name, blob_path = gcs_path.split("/", 1)
            
            logger.info(f"🪣 버킷: {bucket_name}, 경로: {blob_path}")
            
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # PDF 다운로드
            pdf_bytes = blob.download_as_bytes()
            logger.info(f"✅ PDF 다운로드 완료: {len(pdf_bytes):,} bytes")
            
            # PDF 파싱
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                raise RuntimeError("PDF에서 텍스트를 추출할 수 없습니다.")
            
            logger.info(f"✅ 텍스트 추출 완료: {len(text):,} 문자, {len(reader.pages)} 페이지")
            return {"resume_text": text, "page_count": len(reader.pages)}
            
        except Exception as e:
            logger.error(f"❌ GCS PDF 처리 실패: {str(e)}")
            raise RuntimeError(f"GCS에서 PDF 다운로드/처리 실패: {str(e)}")
    
    # 3. HTTP(S) URL에서 PDF 다운로드 (Presigned URL 등 - 레거시 지원)
    if file_path and (file_path.startswith("http://") or file_path.startswith("https://")):
        if not PdfReader:
            raise RuntimeError("pypdf가 설치되지 않았습니다.")
        try:
            logger.info(f"📥 URL에서 PDF 다운로드 중: {file_path[:80]}...")
            response = requests.get(file_path, timeout=60)
            response.raise_for_status()
            
            pdf_bytes = response.content
            logger.info(f"✅ PDF 다운로드 완료: {len(pdf_bytes):,} bytes")
            
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            if not text:
                raise RuntimeError("PDF에서 텍스트를 추출할 수 없습니다.")
            
            logger.info(f"✅ 텍스트 추출 완료: {len(text):,} 문자, {len(reader.pages)} 페이지")
            return {"resume_text": text, "page_count": len(reader.pages)}
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"URL에서 PDF 다운로드 실패: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"PDF 처리 중 오류 발생: {str(e)}")

    if file_path:
        if not PdfReader:
            raise RuntimeError("pypdf가 설치되지 않았습니다.")
        try:
            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                if not text:
                    raise RuntimeError("PDF에서 텍스트를 추출할 수 없습니다.")
                return {"resume_text": text, "page_count": len(reader.pages)}
        except Exception as e:
            raise RuntimeError(f"파일 읽기 중 오류 발생: {str(e)}")
    
    if fallback_text:
        text = fallback_text.strip()
        if not text:
            raise RuntimeError("입력된 텍스트가 비어있습니다.")
        return {"resume_text": text, "page_count": 1}
    
    raise RuntimeError("PDF, 파일 경로, 또는 텍스트 중 하나는 필수입니다.")

load_resume_tool = FunctionTool(func=load_resume_content)


# --- Tool 2: Company Research Request (Google Search는 Agent가 직접 수행) ---
def request_company_research(
    company_name: str,
    search_type: str = "overview"
) -> Dict[str, Any]:
    """기업 조사 요청 정보를 반환합니다. 실제 웹 검색은 Agent의 Google Search Grounding이 수행합니다.
    
    Args:
        company_name: 조사할 기업명
        search_type: 조사 유형 (overview, talent_philosophy, core_values, vision, business)
        
    Returns:
        Dict containing research request information
    """
    
    logger.info(f"🔍 기업 조사 요청: {company_name} - {search_type}")
    
    # 검색 가이드 메시지
    research_guides = {
        "overview": f"{company_name}의 전반적인 정보 (인재상, 핵심가치, 비전, 사업분야)",
        "talent_philosophy": f"{company_name}의 인재상과 채용 정보",
        "core_values": f"{company_name}의 핵심 가치와 기업 문화",
        "vision": f"{company_name}의 비전, 미션, 경영 철학",
        "business": f"{company_name}의 주요 사업 분야와 제품/서비스"
    }
    
    guide = research_guides.get(search_type, f"{company_name}에 대한 정보")
    
    return {
        "company_name": company_name,
        "search_type": search_type,
        "research_guide": guide,
        "instruction": f"웹에서 '{guide}'를 검색하여 최신 정보를 수집해주세요. 공식 홈페이지와 신뢰할 수 있는 출처를 우선적으로 참고하세요.",
        "status": "ready_for_search"
    }

company_research_tool = FunctionTool(func=request_company_research)


# --- Tool 2-1: Search Google (더미 도구 - 모델이 명시적으로 검색 수행) ---
def search_google(query: str) -> str:
    """
    Google 검색 도구 - 모델이 이 함수를 호출하여 웹 검색을 수행합니다.
    실제 검색은 Gemini의 내장 Google Search Grounding이 자동으로 수행합니다.
    
    Args:
        query: 검색 쿼리
        
    Returns:
        검색 가이드 메시지
    """
    logger.info(f"🔍 Google 검색 요청: {query}")
    return f"'{query}'에 대한 웹 검색을 수행하세요. Google Search Grounding을 사용하여 최신 정보를 찾고, 공식 출처 URL을 반드시 포함하세요."

search_google_tool = FunctionTool(func=search_google)




# --- Tool 4: Save Resume Analysis (자기소개서 분석 결과 저장) ---
def save_resume_analysis(
    summary: str,
    experiences: List[Dict[str, Any]],
    technical_skills: List[str],
    soft_skills: List[str],
    achievements: List[str],
    interests: List[str],
    personality_traits: List[str],
    keywords: List[str],
    company_name: str,
    session_id: str
) -> Dict[str, Any]:
    """자기소개서 분석 결과를 구조화하여 GCS에 저장합니다.
    
    Args:
        summary: 핵심 요약 (2-3문장)
        experiences: 주요 경험 리스트 [{"title": "프로젝트명", "description": "설명", "achievements": "성과", "skills_used": [...]}]
        technical_skills: 기술 역량 리스트 (예: ROS, Python, C++)
        soft_skills: 소프트 스킬 리스트 (예: 팀워크, 문제해결)
        achievements: 주요 성과 리스트
        interests: 관심 분야 리스트
        personality_traits: 성격/가치관 리스트
        keywords: 핵심 키워드 리스트
        company_name: 지원 기업명
        session_id: 세션 ID (예: session_20251107_160000)
        
    Returns:
        Dict confirming data was saved with file path
    """
    
    # ✅ 파일명에 세션 ID 사용
    gcs_filename = f"{session_id}_analysis.json"
    
    # 저장할 데이터 구조 (구조화된 분석 결과)
    analysis_data = {
        "sessionId": session_id,
        "company_name": company_name,
        "timestamp": datetime.now().isoformat(),
        "resume_analysis": {
            "summary": summary,
            "experiences": experiences,
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "achievements": achievements,
            "interests": interests,
            "personality_traits": personality_traits,
            "keywords": keywords
        },
        "company_info": None,  # 나중에 company_researcher가 업데이트
        "created_by": "interview_agent_adk"
    }
    
    # GCS에 저장
    gcs_uri = save_to_gcs(analysis_data, gcs_filename, folder="interview_questions")
    
    if gcs_uri is None:
        logger.error(f"❌ GCS 저장 실패!")
        return {
            "status": "error",
            "message": "GCS 저장에 실패했습니다.",
            "gcs_uri": None,
            "filename": None
        }
    
    logger.info(f"✅ 자기소개서 분석 결과 저장 완료: {gcs_filename}")
    logger.info(f"   세션 ID: {session_id}")
    
    return {
        "status": "success",
        "message": f"자기소개서 분석이 완료되고 GCS에 저장되었습니다.",
        "sessionId": session_id,
        "gcs_uri": gcs_uri,
        "filename": gcs_filename
    }

save_resume_analysis_tool = FunctionTool(func=save_resume_analysis)


# --- Tool 5: Update Company Info (기업 정보 업데이트) ---
def update_company_info(
    session_id: str,
    talent_philosophy: List[str],
    core_values: List[str],
    vision: str,
    business_areas: List[str]
) -> Dict[str, Any]:
    """자기소개서 분석 파일에 기업 정보를 추가합니다.
    
    Args:
        session_id: 세션 ID (예: session_20251107_160000)
        talent_philosophy: 인재상 리스트
        core_values: 핵심 가치 리스트
        vision: 비전/미션
        business_areas: 사업 분야 리스트
        
    Returns:
        Dict confirming update was successful
    """
    
    if not storage_client:
        return {
            "status": "error",
            "message": "GCS 클라이언트가 초기화되지 않았습니다."
        }
    
    try:
        # ✅ 세션 ID로 파일명 생성
        filename = f"{session_id}_analysis.json"
        
        # GCS에서 기존 파일 로드
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_path = f"interview_questions/{filename}"
        blob = bucket.blob(blob_path)
        
        if not blob.exists():
            logger.error(f"❌ 파일을 찾을 수 없습니다: {blob_path}")
            return {
                "status": "error",
                "message": f"파일을 찾을 수 없습니다: {filename}"
            }
        
        # 기존 데이터 로드
        existing_data = json.loads(blob.download_as_text())
        
        # 기업 정보 추가
        existing_data["company_info"] = {
            "talent_philosophy": talent_philosophy,
            "core_values": core_values,
            "vision": vision,
            "business_areas": business_areas
        }
        existing_data["updated_at"] = datetime.now().isoformat()
        
        # 업데이트된 데이터 저장
        blob.upload_from_string(
            json.dumps(existing_data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"✅ 기업 정보 업데이트 완료: {filename}")
        
        return {
            "status": "success",
            "message": "기업 정보가 추가되었습니다.",
            "gcs_uri": f"gs://{GCS_BUCKET_NAME}/{blob_path}",
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"❌ 기업 정보 업데이트 실패: {str(e)}")
        return {
            "status": "error",
            "message": f"업데이트 실패: {str(e)}"
        }

update_company_info_tool = FunctionTool(func=update_company_info)


# === AGENTS ===

# --- Single Root Agent (통합 버전 - 속도 최적화) ---
root_agent = Agent(
    name="multi_agent_interview_system",
    model="gemini-2.5-flash",
    description="AI 면접 시스템을 총괄 관리하는 코디네이터 에이전트 (Gemini Function Calling 기반)",
    instruction="""
🎯 **똑터뷰 AI 면접 준비 시스템에 오신 것을 환영합니다!** 🎯

당신은 자기소개서와 기업 정보를 분석하여 면접 데이터를 준비하는 전문가입니다.

**⚡ 중요: 모든 작업을 직접 수행하세요! (Sub-agent 전달 금지)**

**🆔 세션 ID 추출 (최우선!):**
메시지에서 세션 ID를 추출하세요:
```
[SESSION_ID: session_20251107_160000]
```
추출한 세션 ID를 모든 함수 호출 시 사용하세요!

**작업 순서 (반드시 순차적으로):**

**1단계: 자기소개서 로드 및 분석**
   a) `load_resume_content()` 함수를 호출하여 자기소개서 텍스트를 로드하세요
      - 메시지에서 "GCS URI: gs://..." 형식의 URI를 찾으세요
      - 이 GCS URI를 `file_path` 인자로 전달하세요
      - 예: `load_resume_content(file_path="gs://interview-data-cosmic-mariner/pdf/session_xxx_resume.pdf")`
   
   b) 자기소개서를 체계적으로 분석하여 다음 정보를 추출하세요:
      - 지원 기업명 (필수!)
      - 핵심 요약 (2-3문장)
      - 주요 경험/프로젝트 (구조화):
        [{"title": "프로젝트명", "description": "설명", "achievements": "성과", "skills_used": [...]}]
      - 기술 역량 (technical_skills): ["ROS", "Python", ...]
      - 소프트 스킬 (soft_skills): ["팀워크", "문제해결", ...]
      - 주요 성과 (achievements): ["성과1", "성과2", ...]
      - 관심 분야 (interests): ["로봇 공학", ...]
      - 성격/가치관 (personality_traits): ["도전적", ...]
      - 핵심 키워드 (keywords): ["로봇", "무인체계", ...]
   
   c) ✅ `save_resume_analysis()` 함수를 호출하여 GCS에 저장하세요:
      ```
      save_resume_analysis(
          summary="...",
          experiences=[...],
          technical_skills=[...],
          soft_skills=[...],
          achievements=[...],
          interests=[...],
          personality_traits=[...],
          keywords=[...],
          company_name="LIG Nex1",
          session_id="session_20251107_160000"  # ← 추출한 세션 ID!
      )
      ```
   
   d) 반환된 **sessionId**를 기억하세요!

**2단계: 기업 정보 웹 검색**
   a) 1단계에서 추출한 **기업명**을 사용하여 웹 검색을 수행하세요:
      - "[기업명] 인재상 채용 공식"
      - "[기업명] 핵심가치 기업문화 공식"
      - "[기업명] 비전 미션 공식"
      - "[기업명] 사업분야 주요사업"
   
   b) 검색 결과에서 다음 정보를 추출하세요:
      - 인재상 (talent_philosophy): 3-5개
      - 핵심 가치 (core_values): 3-5개
      - 비전/미션 (vision): 1개
      - 사업 분야 (business_areas): 2-4개
   
   c) ✅ `update_company_info()` 함수를 호출하여 기업 정보를 추가하세요:
      ```
      update_company_info(
          session_id="session_20251107_160000",  # ← 추출한 세션 ID!
          talent_philosophy=[...],
          core_values=[...],
          vision="...",
          business_areas=[...]
      )
      ```

**3단계: 완료 메시지**
   "✅ 자기소개서 분석 및 기업 정보 수집 완료! GCS에 저장되었습니다."

**🚨 절대 금지:**
- ❌ transfer_to_agent 사용 금지
- ❌ sub-agent에게 작업 위임 금지
- ❌ 당신이 직접 모든 함수를 호출하세요!

**📋 최종 데이터 구조:**
```json
{
  "company_name": "LIG Nex1",
  "resume_analysis": {
    "summary": "...",
    "experiences": [...],
    "technical_skills": [...],
    "soft_skills": [...],
    "achievements": [...],
    "interests": [...],
    "personality_traits": [...],
    "keywords": [...]
  },
  "company_info": {
    "talent_philosophy": [...],
    "core_values": [...],
    "vision": "...",
    "business_areas": [...]
  }
}
```

시작할 준비가 되셨나요? 📄✨
""",
    sub_agents=[],  # ✅ Sub-agent 제거
    tools=[
        load_resume_tool,
        save_resume_analysis_tool,
        search_google_tool,
        company_research_tool,
        update_company_info_tool
    ],
    include_contents="default",
)