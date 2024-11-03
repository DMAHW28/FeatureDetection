import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from harris import harris_detector, non_maximum_suppression, det_corners_flat_edges, display
IMAGE_SIZE = 300


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.image_file_name = None
        self.master = master
        self.pack(fill="both", expand=True, padx=10, pady=10)
        self.master.title("Reconnaissance d’images et vision par ordinateur")
        self.master.geometry("900x700")

        # Barre de menu
        menubar = tk.Menu(self.master)
        menu_fichier = tk.Menu(menubar, tearoff=0)
        menu_fichier.add_command(label="New")
        menu_fichier.add_command(label="Open", command=lambda: messagebox.showinfo("Ouvrir", "Ouverture du fichier"))
        menu_fichier.add_separator()
        menu_fichier.add_command(label="Quit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=menu_fichier)
        self.master.config(menu=menubar)

        # Onglets
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Onglet "Harris Estimator"
        harris_estimator = ttk.Frame(self.notebook)
        self.notebook.add(harris_estimator, text="Harris Estimator")

        # Cadre pour les paramètres
        param_frame = tk.Frame(harris_estimator)
        param_frame.pack(pady=10, fill="x", padx=20)

        # Sélection d'image
        self.file_label = tk.Label(param_frame, text="No File Selected")
        self.file_label.grid(row=0, column=0, sticky="w", pady=5)
        file_button = tk.Button(param_frame, text="Choose Picture", command=self.choose_image)
        file_button.grid(row=1, column=0, pady=5)

        # Choix de la fenêtre (RadioButtons)
        radio_label = tk.Label(param_frame, text="Choose a Window:", font=("Arial", 12))
        radio_label.grid(row=2, column=0, sticky="w", pady=5)

        self.radio_var = tk.StringVar(value="Gaussian")
        self.rect_estimator = tk.Radiobutton(param_frame, text="Rectangular", variable=self.radio_var,
                                             value="Rectangular", command=self.toggle_sigma)
        self.gaussian_estimator = tk.Radiobutton(param_frame, text="Gaussian", variable=self.radio_var,
                                                 value="Gaussian", command=self.toggle_sigma)
        self.rect_estimator.grid(row=3, column=0, sticky="w")
        self.gaussian_estimator.grid(row=4, column=0, sticky="w")

        # Scale k
        scale_label_k = tk.Label(param_frame, text="k :", font=("Arial", 12))
        scale_label_k.grid(row=5, column=0, sticky="w", pady=5)
        self.scale_k = tk.Scale(param_frame, from_=0.04, to=0.06, orient="horizontal", resolution=0.005)
        self.scale_k.grid(row=6, column=0, sticky="ew", pady=5)

        # Scale kernel
        scale_label_kernel = tk.Label(param_frame, text="Kernel size:", font=("Arial", 12))
        scale_label_kernel.grid(row=7, column=0, sticky="w", pady=5)
        self.scale_kernel = tk.Scale(param_frame, from_=3, to=11, orient="horizontal", resolution=2)
        self.scale_kernel.grid(row=8, column=0, sticky="ew", pady=5)

        # Scale threshold
        scale_label_threshold = tk.Label(param_frame, text="Threshold:", font=("Arial", 12))
        scale_label_threshold.grid(row=9, column=0, sticky="w", pady=5)
        self.scale_threshold = tk.Scale(param_frame, from_=0, to=1, orient="horizontal", resolution=0.005)
        self.scale_threshold.grid(row=10, column=0, sticky="ew", pady=5)

        # Checkbutton NMS
        self.check_nms = tk.BooleanVar()
        checkbutton = tk.Checkbutton(param_frame, text="NMS", variable=self.check_nms)
        checkbutton.grid(row=11, column=0, sticky="w", pady=5)

        # Scale sigma (initialement masqué)
        self.scale_label_sigma = tk.Label(param_frame, text="Sigma:", font=("Arial", 12))
        self.scale_sigma = tk.Scale(param_frame, from_=0, to=1, orient="horizontal", resolution=0.05)

        # Bouton "Compute"
        run_harris = tk.Button(param_frame, text="Compute", font=("Arial", 12, "bold"), command=self.compute_harris_detector)
        run_harris.grid(row=14, column=0, pady=10)

        # Appel initial pour afficher/masquer Sigma
        self.toggle_sigma()

        # Cadre pour les images avec barre de défilement
        image_frame = tk.Frame(harris_estimator)
        image_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Canvas et barres de défilement
        self.canvas = tk.Canvas(image_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        h_scroll = tk.Scrollbar(image_frame, orient="horizontal", command=self.canvas.xview)
        h_scroll.pack(side="bottom", fill="x")
        self.canvas.config(xscrollcommand=h_scroll.set)

        v_scroll = tk.Scrollbar(image_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.pack(side="right", fill="y")
        self.canvas.config(yscrollcommand=v_scroll.set)

        # Frame pour contenir les images
        self.image_container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")

        # Labels pour les images d'origine et traitée
        self.image_label_real = tk.Label(self.image_container)
        self.image_label_real.grid(row=0, column=0, padx=10)

        self.image_label_processed = tk.Label(self.image_container)
        self.image_label_processed.grid(row=0, column=1, padx=10)

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        if file_path:
            self.image_file_name = file_path
            self.file_label.config(text=file_path)
            self.display_image_from_file(file_path)

    def display_image_from_file(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label_real.config(image=img_tk)
        self.image_label_real.image = img_tk

    def display_image(self, img):
        pil_image = Image.fromarray(img)
        pil_image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
        img_tk = ImageTk.PhotoImage(pil_image)
        self.image_label_processed.config(image=img_tk)
        self.image_label_processed.image = img_tk

    def toggle_sigma(self):
        if self.radio_var.get() == "Gaussian":
            self.scale_label_sigma.grid(row=12, column=0, sticky="w", pady=5)
            self.scale_sigma.grid(row=13, column=0, sticky="ew", pady=5)
        else:
            self.scale_label_sigma.grid_forget()
            self.scale_sigma.grid_forget()

    def compute_harris_detector(self):
        if not self.image_file_name:
            messagebox.showinfo("Run", "No picture selected")
        else:
            C = harris_detector(self.image_file_name, window_type=self.radio_var.get(), window_size=self.scale_kernel.get(), k=self.scale_k.get(), sigma=self.scale_sigma.get())
            corners_rect, edges_rect, flat_rect = det_corners_flat_edges(C, threshold=self.scale_threshold.get())
            if self.check_nms.get():
                corners_rect, edges_rect, flat_rect = non_maximum_suppression(C, corners_rect, edges_rect, flat_rect)
            img = display(self.image_file_name, corners_rect, edges_rect, flat_rect)
            self.display_image(img)

        self.image_container.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

