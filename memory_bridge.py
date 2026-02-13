
import os
import logging
import uvicorn
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_bridge")

try:
    from core.redshifted_recall import redshifted_recall
    from core.hallucination_filter import HallucinationDetector
    from core.text_sanitizer import get_sanitizer
except ImportError:
    redshifted_recall = None
    HallucinationDetector = None
    get_sanitizer = None

app = FastAPI(title="OpenClaw Groq Memory Bridge")

GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TARGET_MODEL = os.getenv("TARGET_MODEL", "openai/gpt-oss-120b")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "19192"))
MEMORY_RECALL_K = int(os.getenv("MEMORY_RECALL_K", "3"))
MEMORY_RECALL_THRESHOLD = float(os.getenv("MEMORY_RECALL_THRESHOLD", "0.3"))
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")

detector = HallucinationDetector() if HallucinationDetector else None
sanitizer = get_sanitizer() if get_sanitizer else None


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content) if content else ""


def _validate_bridge_api_key(request: Request):
    if BRIDGE_API_KEY:
        provided_key = request.headers.get("X-API-Key")
        if not provided_key or provided_key != BRIDGE_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "memory_bridge"}


@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    _validate_bridge_api_key(request)

    try:
        payload = await request.json()
    except Exception as e:
        logger.error("Bad request body: %s", e)
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = _extract_text(m.get("content", ""))
            break

    recall_query = sanitizer.strip_meta(last_user_msg) if sanitizer else last_user_msg
    valid_memories = []
    if redshifted_recall and recall_query:
        try:
            results = redshifted_recall(recall_query, k=MEMORY_RECALL_K, threshold=MEMORY_RECALL_THRESHOLD)
            for r in results:
                text = r.get("text", "")
                score = r.get("decayed_score", 0)
                if not text:
                    continue
                if detector:
                    verdict = detector.detect(text)
                    if verdict.get("confidence", 0) >= 0.6:
                        continue
                snippet = text[:200].rstrip()
                valid_memories.append(f"[{score}] {snippet}")
        except Exception as e:
            logger.error("Recall failed: %s", e, exc_info=True)

    if valid_memories:
        injection = "\n[MEM]\n" + "\n".join(valid_memories)
        for m in reversed(messages):
            if m.get("role") == "user":
                if isinstance(m.get("content"), str):
                    m["content"] += injection
                elif isinstance(m.get("content"), list):
                    m["content"].append({"type": "text", "text": injection})
                else:
                    m["content"] = str(m.get("content", "")) + injection
                break

    if valid_memories:
        logger.info("=== MEMORY-AUGMENTED PROMPT === memories_count=%d, query_length=%d", 
                    len(valid_memories), len(recall_query))
    else:
        logger.info("=== NO MEMORIES MATCHED === query: %s", recall_query[:120])

    is_streaming = payload.get("stream", False)
    payload["model"] = TARGET_MODEL
    payload["messages"] = messages
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if is_streaming:
                async with client.stream("POST", GROQ_API_URL, headers=headers, json=payload) as response:
                    async def _stream():
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                    return StreamingResponse(
                        _stream(),
                        status_code=response.status_code,
                        media_type=response.headers.get("content-type", "text/event-stream"),
                    )

            response = await client.post(GROQ_API_URL, headers=headers, json=payload)

            try:
                body = response.json()
            except ValueError:
                logger.error("Groq returned non-JSON (status %s): %s", response.status_code, response.text[:500])
                return JSONResponse(content={"error": "Upstream returned non-JSON", "detail": response.text[:300]}, status_code=502)
            if response.status_code != 200:
                logger.warning("Groq returned %s: %s", response.status_code, str(body)[:300])
            return JSONResponse(content=body, status_code=response.status_code)
    except httpx.TimeoutException:
        logger.error("Groq request timed out")
        return JSONResponse(content={"error": "Upstream timeout"}, status_code=504)
    except Exception as e:
        logger.error("Groq proxy error: %s", e, exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=502)


if __name__ == "__main__":
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable is not set. "
                      "Copy .env.example to .env and add your key.")
        raise SystemExit(1)
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT)
