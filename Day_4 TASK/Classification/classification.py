from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os
from dotenv import load_dotenv

def classify_image(image_path, prediction_client, project_id, model_name):
    try:
        with open(image_path, "rb") as image_data:
            results = prediction_client.classify_image(project_id, model_name, image_data.read())

            # Display the predictions
            print(f"Predictions for image '{os.path.basename(image_path)}':")
            for prediction in results.predictions:
                print(f"- {prediction.tag_name}: {prediction.probability:.2%}")

    except Exception as ex:
        print(f"Error classifying image '{os.path.basename(image_path)}': {ex}")

def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        prediction_endpoint = os.getenv('PredictionEndpoint')
        prediction_key = os.getenv('PredictionKey')
        project_id = os.getenv('ProjectID')
        model_name = os.getenv('ModelName')

        # Authenticate a client for the prediction API
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

        # List of image paths to classify
        image_paths = [
            r'D:\Internship_project\Day_4 TASK\Classification\test_image\apple fruit87 - Copy.jpg',
            r'D:\Internship_project\Day_4 TASK\Classification\test_image\banana85.jpg',
            r'D:\Internship_project\Day_4 TASK\Classification\test_image\orange8.jpg'
        ]

        # Classify multiple test images
        for image_path in image_paths:
            classify_image(image_path, prediction_client, project_id, model_name)
            print()

    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()
