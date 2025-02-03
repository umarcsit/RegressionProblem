# syn-practices-regressions-models
Regression models for Top/Bottom practices and key patterns

This repository contains code for running regression models and generating results, including graphs and error metrics. Follow the steps below to set up and run the code.

* Prerequisites
Before running the code, ensure you have the following installed:
Pip: Pip is required to install the necessary Python packages.
Installation
*Clone the Repository:
bash
git clone https://github.com/Synapbox/syn-practices-regressions-models.git
cd syn-practices-regressions-models
* Install Requirements:
Install the required Python packages by running:
**    pip install -r requirements.txt
Configuration
Enable Models:

Open the indicator.json file.

Set the value of the models you want to run to true. For example:
* {
    "poly_org":true,
    "poly_cm":false,
    "poly_fwm":false,
    "poly_asw":true,
    "rnn_org":false,
    "rnn_cm":false,
    "rnn_fwm":false
}
* Running the Code
Execute the Script:
Run the main script to execute the regression models:

bash
python main.py
Results:

The results, including graphs, will be stored in the Results folder.

Data points and error results will be saved in the datapoints and error_results folders, respectively.

Folder Structure
Results: Contains the generated graphs and visualizations.

datapoints: Stores the data points used in the analysis.

error_results: Contains files with error metrics and results.
