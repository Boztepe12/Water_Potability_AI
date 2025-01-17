import pandas as pd
from tkinter import Tk, Label, Entry, Button, messagebox, Toplevel
from tkinter import ttk
import joblib
import json
import matplotlib.pyplot as plt

class WaterPotabilityApp:
    def __init__(self, master):
        self.master = master
        master.title("Water Potability Prediction")
        master.geometry("400x600")  # Set the window size

        self.features = ['ph', 'Sulfate', 'Hardness', 'Solids', 'Turbidity']
        self.entries = {}

        # Use ttk for modern widgets
        for feature in self.features:
            label = ttk.Label(master, text=feature)
            label.pack(pady=5)
            entry = ttk.Entry(master)
            entry.pack(pady=5)
            self.entries[feature] = entry

        self.predict_button = ttk.Button(master, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = ttk.Label(master, text="")
        self.result_label.pack(pady=10)

        self.graph_button = ttk.Button(master, text="Show Metrics Graphs", command=self.show_graphs)
        self.graph_button.pack(pady=10)

        self.model = joblib.load('random_forest_model.pkl')

    def predict(self):
        try:
            input_data = [float(self.entries[feature].get()) for feature in self.features]
            input_df = pd.DataFrame([input_data], columns=self.features)
            prediction = self.model.predict(input_df)

            if prediction[0] == 1:
                result_text = "The water is drinkable."
            else:
                result_text = "The water is not drinkable."

            self.result_label.config(text=result_text)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")
        except FileNotFoundError:
            messagebox.showerror("Model Error", "Model file not found. Please train the model first.")

    def show_graphs(self):
        metrics_filename = 'metrics.json'

        try:
            with open(metrics_filename, 'r') as f:
                metrics = json.load(f)

            accuracy = metrics['accuracy']
            precision = metrics['precision']
            f1 = metrics['f1_score']
            mse = metrics['mean_squared_error']
            tp = metrics['true_positives']
            fn = metrics['false_negatives']
            tn = metrics['true_negatives']
            fp = metrics['false_positives']

            # Create a new window for metrics
            metrics_window = Toplevel(self.master)
            metrics_window.title("Model Metrics")
            metrics_window.geometry("400x400")  # Set the window size

            # Display accuracy, precision, F1 score, and MSE as text
            accuracy_label = ttk.Label(metrics_window, text=f"Accuracy: {accuracy:.2f}")
            accuracy_label.pack(pady=5)

            precision_label = ttk.Label(metrics_window, text=f"Precision: {precision:.2f}")
            precision_label.pack(pady=5)

            f1_label = ttk.Label(metrics_window, text=f"F1 Score: {f1:.2f}")
            f1_label.pack(pady=5)

            mse_label = ttk.Label(metrics_window, text=f"Mean Squared Error: {mse:.2f}")
            mse_label.pack(pady=5)

            # Display confusion matrix as a graph
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.bar(['TP', 'FN', 'TN', 'FP'], [tp, fn, tn, fp])
            ax.set_title('Confusion Matrix')
            plt.tight_layout()

            # Embed the plot in the Tkinter window
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=metrics_window)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Metrics Error", str(e))

if __name__ == "__main__":
    root = Tk()
    app = WaterPotabilityApp(root)
    root.mainloop()