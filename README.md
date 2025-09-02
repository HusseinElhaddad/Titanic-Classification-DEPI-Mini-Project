# Titanic Classification – DEPI Mini Project  

This project explores the Titanic dataset to predict passenger survival.  
It includes data preprocessing, feature engineering, and a comparison between Logistic Regression and Decision Tree models to evaluate performance.  

<img width="1920" height="577" alt="image" src="https://github.com/user-attachments/assets/9e70159f-4f9c-4920-9d03-3df36b9ef205" />

## Project Workflow  

### 1. Preprocessing  
- **Encoding:** Converted categorical variables (e.g., Sex, Embarked) into numerical values using label encoding and one-hot encoding.  
- **Outliers:** Detected and handled outliers in numerical columns such as Fare and Age.  
- **Dealing with Nulls:** Filled missing Age values using median and filled missing Embarked values with the mode.  
- **Normalization/Standardization:** Scaled numerical features (Age, Fare) to bring them to the same range.  
- **Feature Engineering:** Created new features like FamilySize (SibSp + Parch + 1) and IsAlone, and extracted titles from Name.  

### 2. Model Selection  
- **Logistic Regression:** Used as a baseline model for binary classification.  
- **Decision Tree:** Trained to capture non-linear patterns and compared with Logistic Regression.  

### 3. Model Performance  
- **Metrics:** Evaluated using Accuracy, Precision, Recall, and F1-score.  
- **Comparison:** Logistic Regression gave stable results with fewer parameters, while Decision Tree captured more complex relationships but was prone to overfitting.  

### 4. Results  
- Logistic Regression achieved solid performance as a simple baseline.  
- Decision Tree improved accuracy on training data but required tuning to generalize well.  
- Overall, Logistic Regression was chosen for its balance of performance and interpretability.  

### 5. Deployment  
The final trained model was exported and deployed on Hugging Face Spaces using Gradio for an interactive demo.  

## Try Our Model
You can try the deployed model directly on Hugging Face:  
👉 [Titanic Classification – DEPI Mini Project](https://huggingface.co/HussienElhaddad/Titanic-Classification-DEPI-Mini-Project)  

## Contributors  

Thanks to the whole team for contributing to this project 💻✨  

<a href="https://github.com/HusseinElhaddad">
  <img src="https://avatars.githubusercontent.com/HusseinElhaddad" width="60" height="60" style="border-radius:50%" />
</a>
<a href="https://github.com/MoBahgat010">
  <img src="https://avatars.githubusercontent.com/MoBahgat010" width="60" height="60" style="border-radius:50%" />
</a>
<a href="https://github.com/MohamedAbdelaiem">
  <img src="https://avatars.githubusercontent.com/MohamedAbdelaiem" width="60" height="60" style="border-radius:50%" />
</a>
<a href="https://github.com/Abdelrehim2001">
  <img src="https://avatars.githubusercontent.com/Abdelrehim2001" width="60" height="60" style="border-radius:50%" />
</a>
<a href="https://github.com/Youssef-ahmed12">
  <img src="https://avatars.githubusercontent.com/Youssef-ahmed12" width="60" height="60" style="border-radius:50%" />
</a>
