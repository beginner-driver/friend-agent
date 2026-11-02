"""테스트는 실제 AI를 부르지 않는다. 가짜 응답(fake)과 서버 메모리 저장소로 실행한다.

실제 AI 호출은 비용이 들고, 느리고, 같은 질문에도 답이 달라서 통과와 실패를 판정할 수 없다.
"""
import os

os.environ.update({"LLM_PROVIDER": "fake", "LLM_API_KEY": "", "STORAGE": "memory", "AUTH": "none"})
# .env 에 Langfuse 키가 있어도 테스트의 가짜 대화가 실제 프로젝트로 전송되지 않게 비운다. 키가 비면 계측이 꺼진다.
os.environ.update({"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""})
os.environ.setdefault("ENV", "test")
