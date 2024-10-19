# Customer Feedback Analysis Dashboard for Air New Zealand

This multipage Panel dashboard prototype retrieves real-time data from [Airline Quality (Skytrax)](https://www.airlinequality.com/airline-reviews/air-new-zealand/) through web scraping. It offers interactive charts and graphs powered by Bokeh, aiming to deliver deeper insights and actionable recommendations for stakeholders.

![image](https://github.com/user-attachments/assets/e722aed6-d710-4431-bbbf-da593ea4c609)

```
.
├── 110m_cultural
├── Procfile
├── README.md
├── dashboard
├── dataset
├── lds-nz-meridional-circuit-boundaries-nzgd2000-SHP
├── notebook
├── requirements.txt
├── runtime.txt
└── static

7 directories, 4 files
```

## Prerequisites
Before running the project, ensure that the following tools and libraries are installed:

### Step 1: Clone the repository.
```
git clone https://github.com/air-new-zealand-customer-feedback-analysis.git
cd air-new-zealand-customer-feedback-analysis
```

### Step 2: Create a Conda environment.
```
conda create --name airnz_dashboard python=3.9
conda activate airnz_dashboard
```

### Step 3: Install required dependencies.
```
pip install -r requirements.txt
```

### Step 4: Install NLTK Data. (Optional)
You can skip this step as the repository already includes NLTK data in the root folder. However, if you wish to download it manually, run the following commands from your terminal:
```
python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
```

## Heroku CLI Installation
The Heroku CLI is essential for managing and deploying your Heroku apps from the command line. Follow these platform-specific instructions to install it:

## macOS
The recommended way to install the Heroku CLI on macOS is via Homebrew:

1. Install Homebrew (if not already installed): Homebrew is a package manager for macOS, and it simplifies the installation of software. If Homebrew is not installed, open the terminal and run:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Follow the on-screen instructions to complete the installation.

2. After Homebrew is installed, use it to install the Heroku CLI by running:
```
brew tap heroku/brew && brew install heroku
```

3. Once the installation is complete, check that the Heroku CLI is properly installed:
```
heroku --version
```

For Windows or Linux OS, refer to the Homebrew Documentation. https://brew.sh/

## Heroku Deployment

### Step 1: Create a new Heroku app. Rename your-app-name.
```
heroku create your-app-name
```

### Step 2: Set up Procfile.
Create a Procfile in the root directory of your project:
```
web: panel serve dashboard.py --port=$PORT --address=0.0.0.0
```

### Step 3: Commit your changes.
Ensure everything is committed to your GitHub repository.
```
git add .
git commit -m "Prepare for Heroku deployment"
```

### Step 4: Deploy to Heroku.
Push the code to Heroku using Git:
```
git push heroku main
```

### Step 5: Open your app in the browser.
```
heroku open
```

Wait for few minutes until you see something like this:
![image](https://github.com/user-attachments/assets/f93b3f8b-091a-4db4-9508-68b5b0b098ac)

# Troubleshooting
If you encounter any issues, check the Heroku logs:
```
heroku logs --tail
```
