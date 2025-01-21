### Predictive Analysis API for Manufacturing Operations


Overview

This project provides a RESTful API for predictive analysis in manufacturing operations. The API predicts machine downtime or production defects using machine learning.

Features

Upload Dataset: Upload a manufacturing-related dataset in CSV format.

Train Model: Train a supervised ML model (Decision Tree) on the dataset.

Make Predictions: Predict machine downtime or defects based on input parameters.

Setup Instructions

Prerequisites

Python 3.8+

pip package manager

Installation

Clone the repository:

git clone [<repository_url>](https://github.com/Gauravx2/Model_prediction-using-FastAPI)


Install required dependencies:

pip install -r requirements.txt

Run the API

Start the FastAPI server:

python app.py

Access the API documentation:

Open your browser and navigate to http://127.0.0.1:8000/docs.

Alternatively, use http://127.0.0.1:8000/redoc for an alternative interface.

Endpoints


2. Train Model

Endpoint: POST /train

Description: Train a Decision Tree model and retrieve performance metrics.

Example Request:

curl -X POST "http://127.0.0.1:8000/train"

Response:

{
    "message": "Model trained successfully",
    "accuracy": 0.85,
    "f1_score": 0.75
}

3. Make Predictions

Endpoint: POST /predict

Description: Predict machine downtime using input parameters.

Request Body:

{
    "Temperature": 80,
    "Run_Time": 120
}

Example Request:

curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"Temperature": 80, "Run_Time": 120}'

Response:

{
    "Downtime": "Yes",
    "Confidence": 0.85
}

Testing

Use Postman or cURL to test endpoints locally.

The API documentation at http://127.0.0.1:8000/docs provides interactive testing capabilities.

Project Structure

<repository_directory>/
|-- Operations/main.py             # Main API script
|-- data.csv            # Synthetic dataset (auto-generated if not uploaded)
|-- requirements.txt    # Python dependencies
|-- README.md           # Documentation

Dependencies

FastAPI: Web framework for the API.

scikit-learn: Machine learning library.

pandas: Data manipulation and analysis.

numpy: Numerical operations.

uvicorn: ASGI server for running FastAPI.
python-multipart

Install all dependencies using the requirements.txt file.

Future Improvements

Add authentication for API endpoints.

Enhance model with more advanced algorithms.

Deploy API to cloud platforms like AWS or Azure.

License

This project is licensed under the MIT License.

Contact

For questions or feedback, feel free to reach out:

Author: GauravX2

Email: gaurav123.beg@gmail.com

