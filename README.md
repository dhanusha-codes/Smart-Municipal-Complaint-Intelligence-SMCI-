# Smart Municipal Complaint Intelligence (SMCI)
### A Geo-Visual Analysis System

A web-based municipal complaint management system that combines machine learning and geographical analysis to help identify, classify, prioritize and monitor road infrastructure complaints.

## About the Project

Road infrastructure problems such as potholes, cracks, broken road edges, open manholes and waterlogging can create safety risks for citizens. Managing these complaints manually can make it difficult to identify repeated reports, determine priority levels and locate areas with a high concentration of defects.

**Smart Municipal Complaint Intelligence (SMCI)** aims to address these challenges through a web application that integrates image classification, duplicate complaint detection, rule-based priority assessment and geographical hotspot detection.

Citizens can submit complaints with road images and location information, while administrators can monitor complaints and analyse reported infrastructure issues through an administrative dashboard.

## Key Features

- **Road Defect Classification:** Uses MobileNetV3-Small to classify uploaded images into six road defect categories.
- **Duplicate Complaint Detection:** Uses image hashing and geographical distance comparison to identify potential duplicate complaints.
- **Priority Assessment:** Calculates complaint priority using predefined factors such as severity, impact, location and time.
- **Geographical Hotspot Detection:** Uses DBSCAN clustering to identify areas with concentrated road defect reports.
- **Complaint Management:** Supports complaint submission, tracking and administrative monitoring.
- **Interactive Map Visualization:** Displays geographical information using Leaflet and OpenStreetMap.

## Road Defect Categories

The image classification model identifies six categories:

1. Broken Edges
2. Construction Zones
3. Cracks
4. Open Manholes
5. Potholes
6. Waterlogged Roads

## Machine Learning Model

The project uses **MobileNetV3-Small**, a lightweight convolutional neural network, for road defect image classification.

### Dataset Preparation

- Original dataset: 1,047 images
- Duplicate images removed using MD5 hashing: 376
- Final unique images: 671
- Training split: 70% (469 images)
- Validation split: 15% (101 images)
- Testing split: 15% (101 images)

### Model Performance

| Evaluation Metric | Result |
|---|---:|
| Test Images | 101 |
| Correct Predictions | 78 |
| Accuracy | 77.23% |
| Macro F1-Score | 71.87% |
| Weighted F1-Score | 76.59% |

An SVM model with an RBF kernel was also implemented as a baseline, achieving a macro F1-score of 66.58%.

## Technologies Used

### Frontend
- React
- TypeScript
- Vite
- Leaflet
- OpenStreetMap

### Backend
- Python
- Flask
- SQLite

### Machine Learning and Data Processing
- PyTorch
- Torchvision
- MobileNetV3-Small
- Scikit-learn
- NumPy
- Pillow

### Algorithms
- MobileNetV3-Small – Image Classification
- MD5 – Exact Duplicate Removal During Dataset Preparation
- Average Hash (aHash) – Complaint Image Similarity
- Haversine Formula – Geographical Distance Calculation
- Rule-Based Weighted Scoring – Complaint Prioritization
- DBSCAN – Geographical Hotspot Detection
- SVM with RBF Kernel – Baseline Classification

## System Workflow

1. A citizen submits a complaint with an image and location details.
2. The uploaded image is processed and classified using MobileNetV3-Small.
3. The system checks for potential duplicate complaints.
4. A priority score is calculated using predefined factors.
5. DBSCAN identifies geographical clusters of reported defects.
6. Complaint information and analysis results are made available through the administrative dashboard.

## Project Structure

The main folders in this repository include:

```text
Smart-Municipal-Complaint-Intelligence/
│
├── dataset/
│   └── indian_road_issues/
│
├── frontend/
│
├── ml/
│
├── reports/
│   └── figures/
│
└── Smart_Municipal_Complaint_Intelligence_Report.docx
```

The frontend, machine learning modules, dataset and project report are maintained in their respective folders.

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/dhanusha-codes/Smart-Municipal-Complaint-Intelligence-SMCI-.git
```

```bash
cd Smart-Municipal-Complaint-Intelligence-SMCI-
```

### 2. Run the Backend

Navigate to the Flask backend directory containing the backend entry-point file.

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the backend dependencies using the project's requirements file:

```bash
pip install -r requirements.txt
```

Start the Flask application using its configured entry-point file.

### 3. Run the Frontend

Open another terminal and navigate to the frontend directory.

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed in the terminal to access the application.

## Project Outcomes

The project demonstrates how machine learning and geographical analysis can be integrated into a municipal complaint management application.

It provides automated initial road defect classification, potential duplicate identification, rule-based prioritization and geographical hotspot visualization to support more organized complaint management.

## Future Improvements

- Expand the dataset with more diverse road defect images.
- Improve classification performance, especially for challenging defect categories.
- Enhance duplicate detection using more advanced image similarity techniques.
- Include additional factors in priority assessment.
- Improve real-time geographical monitoring.
- Explore cloud deployment and mobile application integration.
## Acknowledgement

We thank our project supervisor, faculty members and SRM University–AP for their guidance and support throughout the development of this project.
