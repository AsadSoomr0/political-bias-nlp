# Political Bias Detection Using NLP

## Overview
This project applies Natural Language Processing (NLP) techniques to classify political news articles as left-leaning or right-leaning. It utilizes the QBias dataset from [AllSides](https://www.allsides.com/headline-roundups) and a pretrained transformer model for baseline performance evaluation (https://huggingface.co/premsa/political-bias-prediction-allsides-mDeBERTa)

## Setup Instructions
1. Clone the repository:
   ```bash
    git clone https://github.com/AsadSoomr0/political-bias-nlp.git
    cd political-bias-nlp

2. Create and activate a virtual environment

3. Install dependencies:

    pip install -r requirements.txt

4. Accessing the AllSides Dataset:

   
    This project uses the QBias Dataset, which contains 21,747 news articles labeled as left, right, or center

    ### Download the Dataset
    - The dataset is sourced from the **AllSides Balanced News Headline Roundups**:  
    [https://www.allsides.com/headline-roundups](https://www.allsides.com/headline-roundups)
    - The original dataset repository is available on GitHub:  
    [https://github.com/irgroup/Qbias](https://github.com/irgroup/Qbias)

    ### Place the CSV in the Project Directory
    After downloading, move the dataset into the project folder and rename it if needed.

    ## Running the Baseline Model
    Once the dataset is in place, you can continue

5. Run preprocessing:

    python preprocess.py

6. Run the baseline model:

    python baseline_model.py

