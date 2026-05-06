# AI Study Buddy

## Local Run

```powershell
pip install -r requirements.txt
# Put your LLM settings in .env once:
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_key_here
# ANSWER_MODEL_NAME=llama-3.3-70b-versatile
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t ai-study-buddy:latest .
docker run -p 8000:8080 --env-file .env ai-study-buddy:latest
```
