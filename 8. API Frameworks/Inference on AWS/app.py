# On AWS, ensure you have FastAPI and Uvicorn installed:
# pip install fastapi uvicorn scikit-learn joblib # (and any other libraries your model needs)

# Run the app with:
# python app.py

# Start the server with:
# uvicorn app:app --reload --host localhost --port 8000

# Make sure TCP port 8000 is open in your EC2 Security Group settings.

# Access the app in your browser at:
# http://<your-ec2-public-ip>:8000

# lsof -i :8000
# kill -9 <PID>

# ===============================================
# Import required libraries
# ===============================================

# FastAPI: lightweight framework to build APIs and simple web pages
from fastapi import FastAPI, Form
# HTMLResponse: allows us to return HTML directly (no templates needed)
from fastapi.responses import HTMLResponse
# Uvicorn: ASGI server to run the FastAPI app
import uvicorn
# Pickle: to load the pre-trained and saved ML model
import pickle


# ===============================================
# Load the trained model
# ===============================================
# We assume the model was saved with: pickle.dump(model, open("model.pkl","wb"))
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# ===============================================
# Initialize FastAPI app
# ===============================================
app = FastAPI(title="AI Model Demo in Production")


# ===============================================
# Home route: shows an HTML form
# ===============================================
@app.get("/", response_class=HTMLResponse)
def home():
    # Minimal HTML form with 4 input fields
    # (adapt to the number of features your model requires)
    return """
    <html>
        <head>
            <title>Prediction Demo</title>
        </head>
        <body>
            <h2>Enter values for prediction</h2>
            <form action="/predict" method="post">
                <label>Feature 1: <input type="number" step="any" name="f1"></label><br><br>
                <label>Feature 2: <input type="number" step="any" name="f2"></label><br><br>
                <label>Feature 3: <input type="number" step="any" name="f3"></label><br><br>
                <label>Feature 4: <input type="number" step="any" name="f4"></label><br><br>
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """


# ===============================================
# Prediction route: processes the form data
# ===============================================
@app.post("/predict", response_class=HTMLResponse)
def predict(
    f1: float = Form(...),  # Capture the value entered in "Feature 1"
    f2: float = Form(...),  # Capture the value entered in "Feature 2"
    f3: float = Form(...),  # Capture the value entered in "Feature 3"
    f4: float = Form(...),  # Capture the value entered in "Feature 4"
):
    # Put all form values into a list
    features = [f1, f2, f3, f4]

    # Make the prediction using the loaded model
    prediction = model.predict([features])

    # Return the result as an HTML page
    return f"""
    <html>
        <head><title>Result</title></head>
        <body>
            <h2>Model Prediction:</h2>
            <p>{prediction[0]}</p>
            <a href="/">Go back to form</a>
        </body>
    </html>
    """


# ===============================================
# Run the app if executed directly
# ===============================================
if __name__ == "__main__":
    # host="0.0.0.0" makes it accessible from outside EC2
    # port=8000 must be opened in EC2 Security Group
    uvicorn.run(app, host="0.0.0.0", port=8000)
