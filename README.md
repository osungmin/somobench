# 산업 에너지 AI 관제센터 v3

한국 특강용 데모입니다. 합성 공정데이터를 사용하지만, API 키를 연결하면 LLM 응답·RAG 검색 후 종합·Agentic AI Tool Calling은 실제로 실행됩니다.

## 주요 기능
- 강의 모드: 기존 EMS → LLM → RAG → Agentic AI를 단계별 ON/OFF
- 실제 운영 모드: 대시보드 + 공정 설비도 + AI Copilot
- 공정 설비도 이상 설비 점멸
- 한글 사내 가상문서 5종을 대상으로 실제 검색
- OpenAI Responses API 기반 실제 LLM 호출
- Agent가 DATA TOOL / RAG TOOL / CALCULATOR를 실제 함수 호출로 선택

## Streamlit 업데이트
기존 GitHub 저장소에 이 폴더 안의 파일을 같은 위치로 업로드하고 Commit 합니다. 기존 Streamlit 앱은 같은 URL에서 자동 재배포됩니다.

## OpenAI API 연결
Streamlit 앱의 Settings → Secrets에 다음을 입력합니다.

```toml
OPENAI_API_KEY = "발급받은_API_키"
OPENAI_MODEL = "gpt-5.6-luna"
```

API 키는 GitHub에 올리지 마세요. `.streamlit/secrets.toml.example`은 형식 예시일 뿐 실제 키를 넣는 파일이 아닙니다.

## RAG 문서
`documents/` 폴더에 모두 한글로 작성된 가상 사내자료가 들어 있습니다. 실제 현장 적용 시 이 자리에 승인된 설비 매뉴얼, SOP, 정비이력, 에너지진단 보고서 등을 연결합니다.

## 주의
절감액 계산에는 교육용 가정값이 사용됩니다. 실제 제어·정비 의사결정에는 현장 검증과 안전·보안 검토가 필요합니다.
