# AI-Powered Resume Screening and Candidate Ranking System

A machine learning project that demonstrates how resume data can be cleaned, explored, modeled, clustered, and ranked to support recruiter decision-making. The project is implemented as a Jupyter Notebook and includes the input dataset, final enriched output dataset, and generated visualization assets.

## Project Overview

This project builds an end-to-end resume screening workflow using Natural Language Processing and classical machine learning. It processes candidate profile attributes such as skills, education, certifications, experience, projects, AI score, salary expectation, recruiter decision, and target job role.

The workflow includes:

- Exploratory data analysis of 1,000 candidate records.
- Data cleaning and missing certification handling.
- Visual analysis of job roles, recruiter decisions, education levels, AI scores, clusters, and candidate scores.
- TF-IDF feature extraction from combined resume text fields.
- Job role prediction using Logistic Regression.
- Candidate segmentation using K-Means clustering.
- Candidate ranking using a weighted scoring formula.
- Export of the final processed dataset for downstream review.

## Repository Structure

```text
Resume_Parser/
+-- AI_Resume_Screening.csv
+-- Final_Resume_Screening_Output.csv
+-- Resume_Screening_Project.ipynb
+-- Resume_Screening_Project-checkpoint.ipynb
+-- README.md
+-- accuracy.png
+-- ai_score_distribution.png
+-- candidate_score_distribution.png
+-- classification_report.png
+-- cluster_distribution.png
+-- confusion_matrix.png
+-- correlation_heatmap.png
+-- dataset_preview.png
+-- education_distribution.png
+-- job_role_distribution.png
+-- missing_values.png
+-- recruiter_decisions.png
+-- top_candidates.png
```

## Dataset

The source dataset is `AI_Resume_Screening.csv`.

It contains 1,000 resume records with the following fields:

- `Resume_ID`
- `Name`
- `Skills`
- `Experience (Years)`
- `Education`
- `Certifications`
- `Job Role`
- `Recruiter Decision`
- `Salary Expectation ($)`
- `Projects Count`
- `AI Score (0-100)`

The project covers four job-role categories:

- AI Researcher
- Cybersecurity Analyst
- Data Scientist
- Software Engineer

Recruiter decision labels in the dataset:

- Hire: 812 records
- Reject: 188 records

## Methodology

### 1. Data Loading and Inspection

The notebook begins by loading the CSV file with pandas and inspecting dataset shape, column names, data types, missing values, duplicate records, and descriptive statistics.

### 2. Data Cleaning

Missing certification values are replaced with `No Certification`, and duplicate rows are removed before modeling and analysis.

### 3. Exploratory Data Analysis

The notebook uses Matplotlib and Seaborn to visualize:

- Job role distribution
- Recruiter decisions
- Education distribution
- AI score distribution
- Candidate score distribution
- Cluster distribution
- Correlation between numeric fields
- Confusion matrix and classification performance

Generated visual outputs are included as PNG files in the repository.

### 4. Text Feature Engineering

A combined text feature is created from:

- `Skills`
- `Education`
- `Certifications`

This text is transformed into numerical features using `TfidfVectorizer`.

### 5. Job Role Classification

A Logistic Regression model is trained to predict candidate job roles from the TF-IDF features.

Recorded model performance:

- Test accuracy: `1.0`
- 5-fold cross-validation scores: `[1.0, 0.995, 1.0, 1.0, 0.995]`
- Average cross-validation accuracy: `0.998`

### 6. Candidate Clustering

K-Means clustering is applied to group candidates into four clusters based on the TF-IDF resume features.

Recorded cluster distribution:

- Cluster 0: 255 candidates
- Cluster 1: 262 candidates
- Cluster 2: 239 candidates
- Cluster 3: 244 candidates

### 7. Candidate Ranking

The notebook computes a final `Candidate Score` using:

```text
Candidate Score = (0.4 * AI Score) + (2 * Experience in Years) + (2 * Projects Count)
```

The score is rounded to two decimal places and used to rank candidates.

## Output

The processed output is saved as `Final_Resume_Screening_Output.csv`.

Additional fields added by the notebook:

- `combined_text`
- `Cluster`
- `Candidate Score`

The final output can be used to review ranked candidates, analyze candidate groups, and support recruiter shortlisting workflows.

## Requirements

Recommended Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud jupyter
```

## How to Run

1. Open a terminal in the project directory:

```bash
cd C:\Resume_Parser_Project
```

2. Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud jupyter
```

3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open and run:

```text
Resume_Screening_Project.ipynb
```

5. After execution, review the generated output file:

```text
Final_Resume_Screening_Output.csv
```

## Key Files

- `Resume_Screening_Project.ipynb`: Main notebook containing data analysis, model training, clustering, scoring, and export logic.
- `AI_Resume_Screening.csv`: Original resume screening dataset.
- `Final_Resume_Screening_Output.csv`: Final processed dataset with candidate clusters and ranking scores.
- `*.png`: Visual outputs generated from the analysis, including accuracy, classification, distribution, clustering, and ranking charts.

## Business Use Case

This project can help recruitment and HR analytics teams:

- Quickly classify candidates by job role.
- Identify high-ranking candidates using a transparent scoring formula.
- Segment similar candidates into clusters.
- Explore hiring patterns, education distribution, AI scores, and candidate profile trends.
- Reduce manual effort during the initial resume screening stage.

## Responsible Use

This project should be used as a decision-support tool, not as a fully automated hiring decision system. Resume screening models can reflect patterns and biases present in historical or synthetic data. Recruiters should validate model outputs, review candidate profiles manually, and ensure compliance with applicable hiring, fairness, privacy, and data protection requirements.

## Conclusion

The project demonstrates a practical AI-assisted resume screening pipeline using TF-IDF, Logistic Regression, K-Means clustering, and candidate ranking. It provides a clear foundation for extending the system into a production-ready recruitment analytics application with model persistence, API integration, dashboarding, and bias monitoring.


