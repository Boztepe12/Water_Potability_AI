# Random Forest Water Potability Project

This project aims to determine the potability of water using a Random Forest model trained on a dataset containing various water quality parameters. The project includes a graphical user interface (GUI) that allows users to input water quality features and receive predictions regarding water potability.

## Project Structure

```
random-forest-water-potability
├── data
│   └── water_potability.csv
├── src
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── gui.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Dataset

The dataset used for training the Random Forest model is located in the `data/water_potability.csv` file. It contains features related to water quality and a target variable indicating whether the water is drinkable.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd random-forest-water-potability
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the GUI application:
   ```
   python src/gui.py
   ```

2. Input the water quality parameters in the GUI and click on the "Predict" button to determine if the water is drinkable.

## Model Training

The Random Forest model is implemented in the `src/model_training.py` file. It includes functions for training the model on the preprocessed data, evaluating its performance, and saving the trained model for future use.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.