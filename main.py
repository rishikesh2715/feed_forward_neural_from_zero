# main.py

from src.train import run_xor_experiment, run_regression_experiment

def main():
    """
    Entry-point script to run both XOR and Regression experiments.
    Displays all results and plots.
    """
    print("🚀 Starting Project 5 Neural Network Experiments\n")

    # XOR classification task
    run_xor_experiment(epochs=2000, 
                       lr=0.1)

    # Regression with 3 hidden units
    regression_path = "data/Proj5Dataset.xlsx"
    run_regression_experiment(regression_path, 
                              hidden_units=3, 
                              epochs=10000, 
                              lr=0.05)    

    # Regression with 20 hidden units
    run_regression_experiment(regression_path, 
                              hidden_units=20, 
                              epochs=10000, 
                              lr=0.05)

    print("\n✅ All experiments completed successfully!")

if __name__ == "__main__":
    main()
