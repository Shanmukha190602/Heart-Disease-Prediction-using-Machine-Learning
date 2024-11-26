**Environments used:** **Google Colab **upto the 2nd phase. For the 3rd phase and final phase **Jupyter notebook** is used.

As part of Machine Learning Laboratory course in MTech Degree , I have done Term project on **Heart Disease Prediction**.

I have taken the dataset from **PTB DataBase** available in the **Physionet** website.

**URL link for the Dataset:** https://physionet.org/content/ptbdb/1.0.0/

The **PTB Diagnostic ECG Database** is an open-access dataset provided by **PhysioNet**, extensively used in ECG signal analysis and heart disease research. It contains detailed information about **ECG recordings** from healthy individuals and patients with various heart conditions, making it invaluable for machine learning, signal processing, and clinical research.

**Contents**

**Records:** The database contains a total of 549 records from 290 patients.

**File Formats:**

**.dat files:** Contain raw ECG signal data.

**.hea files:** Header files with metadata such as patient ID, diagnostic information, sampling frequency, etc.

**.atr files:** Annotation files that include signal markers and labels.

**Signal Type:**

12-lead ECG signals for each record, captured using standard medical equipment.

**Includes leads:** I, II, III, aVL, aVR, aVF, V1-V6.

**Sampling Frequency:** 1000 Hz (high-resolution ECG recordings).


  **PROJECT PHASE 1**

In the first pahse of the project, I have done how to read raw ECG data from **.dat files** by writing python code for it.

**Steps followed during the project by beginning with reading raw ecg data:** 

**Step 1:** Download the dataset from https://physionet.org/content/ptbdb/1.0.0/ and unzip the file.

**Step 2:** That unzipped files will be in a folder. Upload that Folder to the **Google drive** as shown in below figure so that iterating each file would be easy.

Unzipped folder name is **Database** and that is uploaded into Google drive as shown in below figure.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/aac93549-891f-4412-bd44-adbe8f63f21c">

**Step 3:** After uploadind the unzipped folder to Google drive. Now the code is written for reading raw ECG data. Necessary library modules have been imported for reading ECG data.

**Code for reading raw ECG data for single patient available here:** https://github.com/Shanmukha190602/Heart-Disease-Prediction-using-Machine-Learning/blob/main/Project%20Phase%201.ipynb

  **PROJECT PHASE 2**

In the second phase, we have plotted ECG signals of all the 549 records and the coordinates of the signals in **csv file** format so that extraction of features would be easy by iterating each csv file.
Also we have also got the patient details like age, sex, reason for admission etc... and saved them into a **csv file** format which is named as**patient_details.csv**. **Reason for admission** is our target variable for predicting whether the patient is having heart disease or not. 

The following figure shows that for each dat file corresponding **ECG Signals** in the form of **PDF** format and coordinates of the same pdf is saved in the form of **csv file** format.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/f42dac99-7320-4947-b79f-e2b0e453d24d">

After saving coordinates of the ECG Signals, now we are extracting the heart related features and saving the features of all 549 records into a **single CSV file** which is named as **extracted_features.csv**

**The excel files of both features and patient details are shown below:**

**Heart Disease Features :** https://github.com/Shanmukha190602/Heart-Disease-Prediction-using-Machine-Learning/blob/main/heart_disease_features.csv

**Patient Details:** https://github.com/Shanmukha190602/Heart-Disease-Prediction-using-Machine-Learning/blob/main/patient_details.csv

**The code for the phase 2 is given in below:**

https://github.com/Shanmukha190602/Heart-Disease-Prediction-using-Machine-Learning/blob/main/Project%20Phase%202.ipynb

After running the code, excel files are also created. 

  **PROJECT PHASE 3**

After **CSV files** are generated , classification is done using SVM Classifier with different classifiers. 

Outcome of this phase is the accuracy and ROC Curve. 

The following figure shows the accuracy of each kernel and the ROC Curve 

<img width="629" alt="image" src="https://github.com/user-attachments/assets/8a5b0670-36d4-48ce-955d-c80964aab7cd">

<img width="629" alt="image" src="https://github.com/user-attachments/assets/f8234abc-e867-4d18-be1e-24dc8b018f64">

As you can see with the **linear** and **rbf** kernel we got **83%** accuracy compared with remaining kernels. 

We have taken binary classification, where 0 is healthy and 1 is heart disease . In this case, we have taken **Myocardial Infarction** as Heart disease. 

Code for the phse 3 is: https://github.com/Shanmukha190602/Heart-Disease-Prediction-using-Machine-Learning/blob/main/Project%20Phase%203.ipynb

**PROJECT FINAL PHASE**

In this final phase, to improve the accuracy we have used **Convolutional Neural Network (CNN)**

**CNN is used because:**

**1.Complex Feature Relationships**
CNNs can automatically detect complex interactions between medical features, such as blood pressure, cholesterol, or ECG readings, without manual feature engineering.

**In Code:**
The dataset is prepared by organizing features into matrices, which the CNN processes through convolutional layers to learn these relationships.
The architecture includes multiple layers to analyze different levels of feature interactions.

**2.Dimensionality Reduction**

Convolutional layers apply filters to reduce the dimensionality of the input while retaining important patterns. This simplifies the data for further processing without losing critical information.

**In Code:**
The CNN architecture uses convolutional layers with specified filter sizes and strides to process input data efficiently.
Pooling layers may also be used to downsample feature maps, further reducing data size.

**3.Feature Hierarchy Learning**

CNNs learn feature hierarchies by identifying simple patterns in initial layers and combining them into more complex patterns in deeper layers. This is crucial for understanding multi-level interactions in medical data.

**In Code:**
The model includes convolutional layers followed by fully connected (dense) layers, allowing the network to combine lower-level patterns (e.g., individual feature effects) into higher-order relationships.

**4.Overfitting Prevention**

Overfitting is a risk in deep learning, especially with smaller datasets. CNNs handle this well through weight sharing in convolutional layers and regularization techniques like dropout.

**In Code:**
Dropout layers are used to randomly deactivate neurons during training, reducing the risk of overfitting.
Training and validation metrics, such as loss and accuracy, are monitored to ensure the model generalizes well.

**5.Sequential/Spatial Data Handling**

If the dataset contains sequential or spatial relationships (e.g., time-series data from ECG readings or structured feature arrays), CNNs are excellent at capturing these patterns.

**In Code:**
Input data is reshaped into a format compatible with CNNs (e.g., 2D or 3D arrays), mimicking spatial structures.
The convolutional filters are applied over these data arrays to identify meaningful spatial or sequential relationships.

**6.Performance Advantage**

CNNs often outperform simpler models like SVMs in tasks involving large datasets or complex patterns, achieving higher accuracy and better generalization.

**In Code:**
The model’s performance is evaluated using ROC curves and AUC scores, demonstrating that CNN is suitable for this task.
Training and test data are split, and metrics are computed to compare the CNN’s classification capabilities.

After using Convolutional Network, accuracy has improved upto **85%** which has been shown in below figure.

<img width="344" alt="image" src="https://github.com/user-attachments/assets/25fbc9cd-a85d-48c0-b644-5f6299cb66c0">


Code for the final phase: https://github.com/Shanmukha190602/Heart-Disease-Prediction-using-Machine-Learning/blob/main/Project%20Final%20Phase.ipynb
