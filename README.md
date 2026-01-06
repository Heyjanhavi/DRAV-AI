# DRAV-AI  
### Medicinal Plant Identification System using Deep Learning

DRAV-AI is an AI-powered web application focused on Ayurved and traditional medicine.  
It combines deep learning and natural language processing to help users identify medicinal plants, ask Ayurveda-related questions, and receive home remedy recommendations.

## Key Features

### Medicinal Plant Identification
- Upload an image of a medicinal plant
- Identify the plant using a trained ResNet50 deep learning model
- Display prediction results with confidence
- Useful for students, researchers, and Ayurveda enthusiasts

### Ayurvedic Chatbot
- Ask Ayurveda-related questions in natural language
- Get informative answers based on Ayurvedic principles
- Designed to assist users in understanding plant uses, properties, and benefits

### Home Remedy Recommendation System
- Suggests Ayurvedi Home Remedies based on user queries
- Focuses on plant-based remedies
- Provides safe, educational guidance (not a medical substitute)

### Performance Visualization
- Accuracy curves
- Confusion matrix
- Model evaluation insights

---

## Technology Stack
- Programming Language: Python  
- Deep Learning: PyTorch  
- Model Architecture: ResNet50 (Transfer Learning)  
- Backend: Flask  
- Frontend: HTML, CSS  
- Visualization: Matplotlib
- AI Chatbot: Google Gemini API
- Version Control: Git & GitHub  

---

## Project Structure

DRAV-AI/
│── app.py
│── resnet50_medicinal.pth
│── resnet50_training.ipynb
│── templates/
│── static/
│── Medicinal_plant_dataset/
│   └── sample/
│── .gitignore
│── README.md


##  How the System Works

### Plant Identification Module
1. User uploads a plant image.
2. Image is preprocessed and passed to the trained ResNet50 model.
3. The model predicts the plant class.
4. Associated Ayurvedic information is displayed.

### Ayurvedic Chatbot Module
1. The user enters an Ayurveda-related question through the web interface.
2. The query is sent to the Google Gemini API for processing.
3. Gemini generates a context-aware response based on Ayurvedic and plant-based knowledge.
4. The response is displayed to the user in a conversational format.


### Remedy Recommendation Module
1. User describes a health concern or symptom.
2. The system suggests suitable plant-based home remedies.
3. Recommendations are informational and educational in nature.


## Model Training
- Transfer learning applied using a pre-trained ResNet50 model.
- Final layers fine-tuned on a medicinal plant dataset.
- Training, evaluation, and performance plots are documented in:


## Dataset Information
- The complete dataset is not included due to size constraints.
- A small sample dataset is provided for demonstration:


