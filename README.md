# 똑터뷰 (Ddockterview) - AI 면접 준비 시스템

AI 기반 맞춤형 면접 질문 생성 및 피드백 시스템

## 📋 시스템 구성

### 1. **질문 생성 Agent** (`interview_agent/`)
- 자기소개서 PDF 분석
- 기업 정보 웹 검색
- 맞춤형 면접 질문 생성

### 2. **면접 진행 Agent** (`session_agent/`)
- 실시간 면접 진행 관리
- 꼬리질문 생성
- 진행 상황 추적

### 3. **피드백 Agent** (`feedback_agent/`)
- 답변 영상 분석 (STT + 행동 분석)
- 언어적/비언어적 피드백 생성
- 최종 점수 리포트 생성

### 4. **API 서버** (`AI_server_cloud_run/`)
- FastAPI 기반 REST API
- Cloud Run 배포용 서버
- 프론트엔드 연동

## 🚀 설치 및 실행

### 1. 환경 설정

#### (1) `.env` 파일 생성
```bash
cp .env.example .env
```

#### (2) `.env` 파일 수정
```env
# Google Cloud 프로젝트 ID 입력
GOOGLE_CLOUD_PROJECT=your-project-id

# GCS 버킷 이름 입력
GCS_BUCKET_NAME=your-bucket-name

# 배포된 Agent ID 입력
SESSION_AGENT_ID=your-session-agent-id
QUESTION_AGENT_ID=your-question-agent-id
```

**⚠️ 중요: `.env` 파일은 절대 GitHub에 업로드하지 마세요!**

### 2. Google Cloud 인증 설정

#### 방법 1: gcloud CLI (로컬 개발)
```bash
gcloud auth application-default login
gcloud config set project your-project-id
```

#### 방법 2: 서비스 계정 키 (배포 환경)
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### 3. 의존성 설치

각 Agent 폴더에서 requirements.txt 설치:

```bash
# Interview Agent
cd interview_agent
pip install -r requirements.txt

# Session Agent
cd ../session_agent
pip install -r requirements.txt

# Feedback Agent
cd ../feedback_agent
pip install -r requirements.txt
```

### 4. Agent 배포 (Google ADK)

```bash
# Interview Agent 배포
cd interview_agent
adk web

# Session Agent 배포
cd ../session_agent
adk web

# Feedback Agent 배포
cd ../feedback_agent
adk web
```

배포 후 생성된 Agent ID를 `.env` 파일에 입력하세요.

## 📁 폴더 구조

```
ddockterview/
├── interview_agent/          # Phase 1: 질문 생성
│   ├── agent.py
│   └── requirements.txt
├── session_agent/            # Phase 2: 면접 진행
│   ├── agent.py
│   └── requirements.txt
├── feedback_agent/           # Phase 3: 피드백 생성
│   ├── agent.py
│   └── requirements.txt
├── AI_server_cloud_run/      # FastAPI 서버
│   ├── interview_router.py   # 면접 진행 API
│   └── question_router.py    # 질문 생성 API
├── .env.example              # 환경변수 템플릿
├── .gitignore                # Git 제외 파일 목록
└── README.md                 # 이 파일
```

## 🔐 보안 주의사항

다음 정보는 **절대 GitHub에 업로드하지 마세요**:

- ❌ `.env` 파일
- ❌ Google Cloud 서비스 계정 키 (`.json` 파일)
- ❌ API 키, 비밀번호
- ❌ 프로젝트 ID, 버킷 이름, Agent ID (하드코딩 금지)

모든 민감 정보는 **환경변수**로 관리하세요!

## 🔧 GCS 버킷 구조

```
your-bucket-name/
├── pdf/                      # 업로드된 자기소개서 PDF
│   └── session_xxx_resume.pdf
├── interview_questions/      # 분석 결과 (질문 데이터)
│   └── session_xxx_analysis.json
├── progress_interview/       # 면접 진행 상황
│   └── session_xxx_progress.json
├── video/                    # 답변 영상
│   └── session_xxx_q1.webm
└── feedback_results/         # 피드백 결과
    ├── session_xxx_q1_feedback.json
    └── session_xxx_final.json
```

## 📝 API 엔드포인트

### 질문 생성
- `POST /api/generate-questions` - 자기소개서 업로드 및 질문 생성

### 면접 진행
- `POST /api/interview/start` - 면접 시작
- `POST /api/interview/upload-answer` - 답변 영상 업로드
- `GET /api/interview/status/{session_id}` - 진행 상황 조회

## 🛠️ 기술 스택

- **AI**: Google Gemini 2.5 Pro/Flash
- **Agent Framework**: Google ADK (Agent Development Kit)
- **Backend**: FastAPI, Python 3.11+
- **Storage**: Google Cloud Storage (GCS)
- **Deployment**: Google Cloud Run
- **TTS**: Google Cloud Text-to-Speech (Gemini-TTS)

## 📞 문의

프로젝트 관련 문의사항이 있으시면 Issues를 통해 연락주세요.

---

**Made with ❤️ for better interview preparation**

