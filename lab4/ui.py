import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import data_processor as dp
import dataset_generator as data_gen
import threading
import os

class DataProcessorAppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processing Tool")

        self.df = None
        self.df_display = None
        self.df_missing = None
        self.file_path = None
        self.categorical_cols = []
        self.mappings = {}
        self.numeric_cols = []

        # --- Left Frame: Controls ---
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        # Dataset Generation
        ttk.Label(self.left_frame, text="Dataset Size:").grid(row=0, column=0, sticky=tk.W)
        self.dataset_size_var = tk.StringVar(value="1000")
        self.count_entry = ttk.Entry(self.left_frame, textvariable=self.dataset_size_var)
        self.count_entry.grid(row=0, column=1, sticky=tk.W)

        # Load Dataset Button
        ttk.Button(self.left_frame, text="Load Dataset", command=self.load_dataset).grid(row=2, column=0, columnspan=2, pady=5)

        # Missing Value Injection
        ttk.Label(self.left_frame, text="Missing %:").grid(row=3, column=0, sticky=tk.W)
        self.missing_percent_var = tk.StringVar(value="30")
        self.missing_entry = ttk.Entry(self.left_frame, textvariable=self.missing_percent_var)
        self.missing_entry.grid(row=3, column=1, sticky=tk.W)

        ttk.Button(self.left_frame, text="Inject Missing", command=self.inject_missing_values).grid(row=4, column=0, columnspan=2, pady=5)

        # Imputation Method Selection
        ttk.Label(self.left_frame, text="Imputation Method:").grid(row=5, column=0, sticky=tk.W)
        self.imputation_method_var = tk.StringVar(value="Hot-Deck")
        imputation_methods = ["Hot-Deck", "Median", "Linear Regression"]
        self.imputation_dropdown = ttk.Combobox(self.left_frame, textvariable=self.imputation_method_var, values=imputation_methods, state="readonly")
        self.imputation_dropdown.grid(row=5, column=1, sticky=tk.W)

        ttk.Button(self.left_frame, text="Impute Missing", command=self.impute_missing_values).grid(row=6, column=0, columnspan=2, pady=5)

        # --- Right Frame: Data Display ---
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.tree = ttk.Treeview(self.right_frame, columns=("col1", "col2"), show="headings")  # Example columns
        self.tree.pack(expand=True, fill=tk.BOTH)

        # --- Configure Grid Weights ---
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Dataset Generation Frame
        self.dataset_generation_frame = ttk.Frame(self.left_frame, padding=10)
        self.dataset_generation_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)

        ttk.Label(self.dataset_generation_frame, text="Filename:").grid(row=0, column=0, sticky=tk.W)
        self.output_entry = ttk.Entry(self.dataset_generation_frame)
        self.output_entry.grid(row=0, column=1, sticky=tk.W)
        self.output_entry.insert(0, "generated_dataset")

        ttk.Label(self.dataset_generation_frame, text="Sberbank%:").grid(row=1, column=0, sticky=tk.W)
        self.a_entry = ttk.Entry(self.dataset_generation_frame)
        self.a_entry.grid(row=1, column=1, sticky=tk.W)
        self.a_entry.insert(0, "25")

        ttk.Label(self.dataset_generation_frame, text="T-Bank%:").grid(row=2, column=0, sticky=tk.W)
        self.b_entry = ttk.Entry(self.dataset_generation_frame)
        self.b_entry.grid(row=2, column=1, sticky=tk.W)
        self.b_entry.insert(0, "25")

        ttk.Label(self.dataset_generation_frame, text="VTB%:").grid(row=3, column=0, sticky=tk.W)
        self.c_entry = ttk.Entry(self.dataset_generation_frame)
        self.c_entry.grid(row=3, column=1, sticky=tk.W)
        self.c_entry.insert(0, "25")

        ttk.Label(self.dataset_generation_frame, text="Alpha-bank%:").grid(row=4, column=0, sticky=tk.W)
        self.d_entry = ttk.Entry(self.dataset_generation_frame)
        self.d_entry.grid(row=4, column=1, sticky=tk.W)
        self.d_entry.insert(0, "25")

        ttk.Label(self.dataset_generation_frame, text="Mastercard%:").grid(row=5, column=0, sticky=tk.W)
        self.x_entry = ttk.Entry(self.dataset_generation_frame)
        self.x_entry.grid(row=5, column=1, sticky=tk.W)
        self.x_entry.insert(0, "33")

        ttk.Label(self.dataset_generation_frame, text="Visa%:").grid(row=6, column=0, sticky=tk.W)
        self.y_entry = ttk.Entry(self.dataset_generation_frame)
        self.y_entry.grid(row=6, column=1, sticky=tk.W)
        self.y_entry.insert(0, "33")

        ttk.Label(self.dataset_generation_frame, text="Mir%:").grid(row=7, column=0, sticky=tk.W)
        self.z_entry = ttk.Entry(self.dataset_generation_frame)
        self.z_entry.grid(row=7, column=1, sticky=tk.W)
        self.z_entry.insert(0, "34")

        ttk.Button(self.dataset_generation_frame, text="Generate Dataset", command=lambda: self.generate_dataset(self.a_entry, self.b_entry, self.c_entry, self.d_entry, self.x_entry, self.y_entry, self.z_entry, self.count_entry, self.output_entry)).grid(row=8, column=0, columnspan=2, pady=5)

    def load_dataset(self):
        self.file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            try:
                self.df = pd.read_csv(self.file_path)
                self.df_display = self.df.copy()
                self.categorical_cols = dp.detect_categorical_columns(self.df)
                self.mappings = dp.create_mappings(self.df, self.categorical_cols)
                self.numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col]) and col not in ['Name', 'Passport', 'Snils', 'Card']]
                self.populate_treeview()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def generate_dataset(self, a_entry, b_entry, c_entry, d_entry, x_entry, y_entry, z_entry, count_entry, output_entry):
        size = int(self.dataset_size_var.get())

        banks_p = {
            "sberbank": int(a_entry.get()),
            "tbank": int(b_entry.get()),
            "vtb": int(c_entry.get()),
            "alphabank": int(d_entry.get()),
        }

        systems_p = {
            "mastercard": int(x_entry.get()),
            "visa": int(y_entry.get()),
            "mir": int(z_entry.get()),
        }
        try:
            data_gen.compute(size, output_entry.get(), banks_p, systems_p)
            self.file_path = output_entry.get() + ".csv"
            self.df = pd.read_csv(self.file_path)
            self.df_display = self.df.copy()
            self.categorical_cols = dp.detect_categorical_columns(self.df)
            self.mappings = dp.create_mappings(self.df, self.categorical_cols)
            self.numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col]) and col not in ['Name', 'Passport', 'Snils', 'Card']]
            self.populate_treeview()
            messagebox.showinfo("Success", "Dataset generated successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate dataset: {e}")

    def inject_missing_values(self):
        if self.df is None:
            messagebox.showinfo("Info", "Load or generate a dataset first.")
            return

        # Define non-modifiable columns
        non_modifiable_cols = ['Passport', 'Snils', 'Card', 'DateStart', 'DateEnd']

        # Columns to modify (excluding 'Name' and non-modifiable columns)
        cols_to_modify = [col for col in self.df.columns if col not in ['Name'] + non_modifiable_cols]
        percent = float(self.missing_percent_var.get()) / 100

        # Apply to the df before and displayed for UI
        self.df_missing = self.df.copy()
        df_missing_injected = dp.inject_missing_values(self.df_missing, percent, cols_to_modify)
        self.df_display = df_missing_injected

        df_mapped = dp.apply_mappings(df_missing_injected, self.mappings)
        self.df = df_mapped

        self.numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col]) and col not in ['Name', 'Passport', 'Snils', 'Card']] #check which are not numeric

        self.populate_treeview()

        messagebox.showinfo("Success", "Missing values injected.")

    def impute_missing_values(self):
        if self.df is None:
            messagebox.showinfo("Info", "Load or generate a dataset first.")
            return

        method = self.imputation_method_var.get()

        df_imputed_mapped = dp.impute_missing_values(self.df, method, self.numeric_cols)

        df_display_reverted = self.df.copy()
        mappings_to_use = self.mappings
        for col in self.categorical_cols:
          if col in mappings_to_use:
            df_display_reverted[col] = df_display_reverted[col].map({v: k for k, v in mappings_to_use[col].items()})

        self.df_display = df_display_reverted

        self.df = df_imputed_mapped

        if self.file_path:
            output_dir = os.path.join(os.path.dirname(self.file_path), "imputed")
            os.makedirs(output_dir, exist_ok=True)

            # Create the output file path within the "imputed" directory
            file_name, file_extension = os.path.splitext(os.path.basename(self.file_path))
            output_path = os.path.join(output_dir, f"{file_name}_imputed_{method}{file_extension}")

            # Save the DataFrame to the output path
            try:
                self.df_display.to_csv(output_path, index=False)
                messagebox.showinfo("Success", f"Missing values imputed using {method}. Dataset saved to {output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save imputed dataset: {e}")
        else:
            messagebox.showinfo("Info", "No file loaded, cant save it. Load or generate a dataset first.")

        # Refresh Treeview
        self.populate_treeview()

    def populate_treeview(self):
        """Populates the Treeview with the DataFrame's data."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.df_display is not None:
            self.tree["columns"] = list(self.df_display.columns)
            self.tree["show"] = "headings"

            for col in self.df_display.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100)

            for index, row in self.df_display.iterrows():
                self.tree.insert("", tk.END, values=list(row))

    @staticmethod
    def is_numeric_dtype_string(dtype):
        """Helper function to check if a dtype string represents a numeric type."""
        numeric_types = ['int', 'float']
        return any(numeric_type in dtype.name for numeric_type in numeric_types)