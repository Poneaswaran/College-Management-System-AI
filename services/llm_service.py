import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag.config import Settings
from rag.errors import AppError


class OllamaLLMService:
    """LLM client with retry, timeout, and connection pooling."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

    def _build_prompt(self, question: str, context_chunks: list[str]) -> str:
        trimmed_context = "\n\n".join(context_chunks)
        return (
            "You are an academic assistant for a College Management System. "
            "Answer only from the retrieved context. If context is insufficient, "
            "state that the material does not contain enough information.\n\n"
            f"Context:\n{trimmed_context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
    )
    async def _generate(self, prompt: str) -> str:
        response = await self.client.post(
            self.settings.OLLAMA_URL,
            json={
                "model": self.settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        answer = str(data.get("response", "")).strip()
        if not answer:
            raise AppError(status_code=503, detail="LLM returned empty response", code="llm_empty")
        return answer

    async def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        prompt = self._build_prompt(question=question, context_chunks=context_chunks)
        try:
            return await self._generate(prompt)
        except AppError:
            raise
        except httpx.TimeoutException as exc:
            raise AppError(
                status_code=503,
                detail="LLM provider timeout",
                code="llm_timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise AppError(
                status_code=503,
                detail="LLM provider unavailable",
                code="llm_provider_error",
            ) from exc

    async def aclose(self) -> None:
        await self.client.aclose()


def create_llm_service(settings: Settings) -> OllamaLLMService:
    provider = settings.LLM_PROVIDER.strip().lower()
    if provider == "ollama":
        return OllamaLLMService(settings)

    raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
