from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root(name: str = "Eugenio"):
    return {
        "message": f"Hello, {name}!"
    }