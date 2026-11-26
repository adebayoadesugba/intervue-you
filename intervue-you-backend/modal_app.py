import sys
import modal

# 1. Define the container environment
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("app", remote_path="/root/app")
    .add_local_dir("vector_db", remote_path="/root/vector_db")
)

app = modal.App("intervue-you-rag")

# 2. Expose your existing FastAPI app as a public web endpoint
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("intervue-secrets")],
    memory=2048,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from app.main import app as web_app
    return web_app