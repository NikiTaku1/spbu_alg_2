import tkinter as tk
from tsp_gui import TSPInteractiveGUI

if __name__ == "__main__":
    root = tk.Tk()
    gui = TSPInteractiveGUI(root)
    root.mainloop()