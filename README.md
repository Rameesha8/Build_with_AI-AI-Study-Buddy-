# AI Study Buddy

## Local Run

```powershell
pip install -r requirements.txt
$env:GEMINI_API_KEY="your_key_here"
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t ai-study-buddy:latest .
docker run -p 8000:8080 -e GEMINI_API_KEY=your_key_here ai-study-buddy:latest
```