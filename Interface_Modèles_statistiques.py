import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class IAInterface:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Projet IA - Modèles Statistiques")
        self.root.geometry("1000x700")
        
        self.bg_color = "#F5ECE0"       # Fond rose très clair
        self.button_color = "#001F54"   # Bleu marine foncé
        self.text_color = "#002244"     # Bleu marine légèrement plus clair pour le texte

        self.create_main_interface()
    
    def create_main_interface(self):
        """Interface principale"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True)
        
        # Titre
        title_frame = tk.Frame(main_frame, bg=self.bg_color)
        title_frame.pack(pady=50)
        
        tk.Label(title_frame, 
                text="Modèles de L'Intelligence Artificielle",
                font=('Helvetica', 20, 'bold'),
                bg=self.bg_color,
                fg=self.text_color).pack()
        
        # Boutons
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(pady=30)
        
        tk.Button(button_frame,
                 text="Entrer",
                 command=self.show_algorithms,
                 bg=self.button_color,
                 fg='white',
                 font=('Helvetica', 14),
                 width=10,
                 height=2).pack(side='left', padx=20)
        
        tk.Button(button_frame,
                 text="Sortie",
                 command=self.root.destroy,
                 bg="#FF6B6B",
                 fg='white',
                 font=('Helvetica', 14),
                 width=10,
                 height=2).pack(side='right', padx=20)
    
    def show_algorithms(self):
        """Interface des algorithmes"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True)
        
        tk.Label(main_frame,
                text="Selectionnez un Modèle",
                font=('Helvetica', 18, 'bold'),
                bg=self.bg_color,
                fg=self.text_color).pack(pady=40)
        
        algorithms = [
            ("Regression Lineaire", self.show_regression_input),
            ("Clustering", self.show_clustering_input),
            ("TimeSeries ARIMA", self.show_arima_input),
            ("Random Forest", self.show_random_forest_input),
            ("Validation Croisee", self.show_cross_validation)
        ]
        
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack()
        
        for i, (text, cmd) in enumerate(algorithms):
            btn = tk.Button(button_frame,
                          text=text,
                          command=cmd,
                          bg=self.button_color,
                          fg='white',
                          font=('Helvetica', 12),
                          width=15,
                          height=2)
            btn.grid(row=i//2, column=i%2, padx=15, pady=15)
        
        # Bouton Retour
        tk.Button(main_frame,
                 text="Retour",
                 command=self.create_main_interface,
                 bg="#AAAAAA",
                 fg='white',
                 font=('Helvetica', 12),
                 width=10,
                 height=1).pack(pady=30)
    
    def show_regression_input(self):
        """Régression linéaire avec labels descriptifs"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(main_frame,
                text="Regression Lineaire - Paramètres de vol",
                font=('Helvetica', 16, 'bold'),
                bg=self.bg_color).pack(pady=10)
        
        # Frame pour les inputs
        input_frame = tk.Frame(main_frame, bg=self.bg_color)
        input_frame.pack(pady=20)
        
        # Input 1: Heures de vol
        tk.Label(input_frame, text="Heures de vol (séparées par virgules):", 
                bg=self.bg_color).grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.reg_x = tk.Entry(input_frame, width=40)
        self.reg_x.grid(row=0, column=1, padx=5, pady=5)
        self.reg_x.insert(0, "100,200,300,400,500,600,700,800,900,1000")
        
        # Input 2: Consommation carburant
        tk.Label(input_frame, text="Consommation carburant (kg, séparées par virgules):", 
                bg=self.bg_color).grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.reg_y = tk.Entry(input_frame, width=40)
        self.reg_y.grid(row=1, column=1, padx=5, pady=5)
        self.reg_y.insert(0, "250,500,750,1000,1250,1500,1750,2000,2250,2500")
        
        # Input 3: Titre du graphique
        tk.Label(input_frame, text="Titre du graphique:", 
                bg=self.bg_color).grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.reg_title = tk.Entry(input_frame, width=40)
        self.reg_title.grid(row=2, column=1, padx=5, pady=5)
        self.reg_title.insert(0, "Consommation carburant vs Heures de vol")
        
        # Boutons
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame,
                 text="Générer",
                 command=self.generate_regression,
                 bg=self.button_color,
                 fg='white',
                 font=('Helvetica', 12)).pack(side='left', padx=10)
        
        tk.Button(button_frame,
                 text="Retour",
                 command=self.show_algorithms,
                 bg="#AAAAAA",
                 fg='white',
                 font=('Helvetica', 12)).pack(side='right', padx=10)
        
        # Frame pour le graphique
        self.reg_graph_frame = tk.Frame(main_frame, bg=self.bg_color)
        self.reg_graph_frame.pack(fill='both', expand=True)
    
    def generate_regression(self):
        """Génère la régression linéaire"""
        try:
            X = np.array([float(x.strip()) for x in self.reg_x.get().split(',')]).reshape(-1, 1)
            Y = np.array([float(y.strip()) for y in self.reg_y.get().split(',')]).reshape(-1, 1)
            
            if len(X) != len(Y):
                messagebox.showerror("Erreur", "Les listes doivent avoir la même longueur")
                return
            
            model = LinearRegression()
            model.fit(X, Y)
            Y_pred = model.predict(X)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X, Y, color='blue', label='Données réelles')
            ax.plot(X, Y_pred, color='red', linewidth=2, label='Régression linéaire')
            ax.set_xlabel("Heures de vol")
            ax.set_ylabel("Consommation carburant (kg)")
            ax.set_title(self.reg_title.get())
            ax.legend()
            ax.grid(True)
            
            for widget in self.reg_graph_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.reg_graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            coef_frame = tk.Frame(self.reg_graph_frame, bg=self.bg_color)
            coef_frame.pack(fill='x', padx=20, pady=(0, 20))
            
            tk.Label(coef_frame, 
                    text=f"Equation: Consommation = {model.coef_[0][0]:.2f} × Heures + {model.intercept_[0]:.2f}",
                    font=('Helvetica', 12),
                    bg=self.bg_color).pack()
            
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides séparés par des virgules")
    
    def show_clustering_input(self):
        """Clustering avec labels descriptifs"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(main_frame,
                text="Clustering K-Means - Paramètres de vol",
                font=('Helvetica', 16, 'bold'),
                bg=self.bg_color).pack(pady=10)
        
        # Frame pour les inputs
        input_frame = tk.Frame(main_frame, bg=self.bg_color)
        input_frame.pack(pady=20)
        
        # Input 1: Vitesse des avions
        tk.Label(input_frame, text="Vitesse des avions (km/h, séparées par virgules):", 
                bg=self.bg_color).grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.clust_x = tk.Entry(input_frame, width=40)
        self.clust_x.grid(row=0, column=1, padx=5, pady=5)
        self.clust_x.insert(0, "200,300,400,500,600,700,800,200,300,400")
        
        # Input 2: Altitude des avions
        tk.Label(input_frame, text="Altitude des avions (m, séparées par virgules):", 
                bg=self.bg_color).grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.clust_y = tk.Entry(input_frame, width=40)
        self.clust_y.grid(row=1, column=1, padx=5, pady=5)
        self.clust_y.insert(0, "5000,6000,7000,8000,9000,5000,6000,7000,8000,9000")
        
        # Input 3: Nombre de clusters
        tk.Label(input_frame, text="Nombre de groupes à créer:", 
                bg=self.bg_color).grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.n_clusters = ttk.Spinbox(input_frame, from_=2, to=10, width=5)
        self.n_clusters.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.n_clusters.set(3)
        
        # Boutons
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame,
                 text="Générer",
                 command=self.generate_clustering,
                 bg=self.button_color,
                 fg='white',
                 font=('Helvetica', 12)).pack(side='left', padx=10)
        
        tk.Button(button_frame,
                 text="Retour",
                 command=self.show_algorithms,
                 bg="#AAAAAA",
                 fg='white',
                 font=('Helvetica', 12)).pack(side='right', padx=10)
        
        # Frame pour le graphique
        self.clust_graph_frame = tk.Frame(main_frame, bg=self.bg_color)
        self.clust_graph_frame.pack(fill='both', expand=True)
    
    def generate_clustering(self):
        """Génère le clustering"""
        try:
            X = np.array([float(x.strip()) for x in self.clust_x.get().split(',')])
            Y = np.array([float(y.strip()) for y in self.clust_y.get().split(',')])
            
            if len(X) != len(Y):
                messagebox.showerror("Erreur", "Les listes doivent avoir la même longueur")
                return
            
            df = pd.DataFrame({'Vitesse (km/h)': X, 'Altitude (m)': Y})
            
            kmeans = KMeans(n_clusters=int(self.n_clusters.get()), random_state=42)
            clusters = kmeans.fit_predict(df)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(df['Vitesse (km/h)'], df['Altitude (m)'], c=clusters, cmap='viridis')
            ax.set_xlabel("Vitesse (km/h)")
            ax.set_ylabel("Altitude (m)")
            ax.set_title(f"Groupes de vols (K={self.n_clusters.get()})")
            ax.grid(True)
            
            for widget in self.clust_graph_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.clust_graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
            
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides séparés par des virgules")

    def show_random_forest_input(self):
        """Affiche le formulaire pour la prédiction Random Forest avion"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(main_frame,
                 text="Random Forest - Prédiction de maintenance avion",
                 font=('Helvetica', 16, 'bold'),
                 bg=self.bg_color).pack(pady=10)
        
        self.rf_result_frame = tk.Frame(main_frame, bg=self.bg_color)
        self.rf_result_frame.pack(pady=10)

        # Formulaire de saisie des paramètres avion
        form_frame = tk.Frame(self.rf_result_frame, bg=self.bg_color)
        form_frame.pack(pady=10)
    
        labels = ["Température Huile (°C)", "Vibrations Moteur", "Pression Carburant (PSI)"]
        self.entries = []
    
        for i, label in enumerate(labels):
            tk.Label(form_frame, text=label, bg=self.bg_color).grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = tk.Entry(form_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries.append(entry)
    
        # Bouton pour lancer la prédiction
        tk.Button(self.rf_result_frame, text="Prédire", command=self.predict_random_forest,
                  bg=self.button_color, fg='white', font=('Helvetica', 12)).pack(pady=10)
        tk.Button(main_frame, text="Retour", command=self.show_algorithms,
                  bg="#AAAAAA", fg='white', font=('Helvetica', 12)).pack(pady=10)
    
        self.result_label = tk.Label(self.rf_result_frame, text="", font=('Helvetica', 14, 'bold'), bg=self.bg_color)
        self.result_label.pack(pady=10)

    def predict_random_forest(self):
        """Entraîne un modèle Random Forest et prédit à partir des entrées utilisateur avec explication"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            import numpy as np
            from tkinter import messagebox
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
            # Supprimer le graphe précédent s'il existe
            if hasattr(self, 'rf_canvas') and self.rf_canvas:
                self.rf_canvas.get_tk_widget().destroy()
    
            # Génération des données synthétiques
            np.random.seed(42)
            n_samples = 300
            X = np.column_stack([
                np.random.normal(85, 5, n_samples),       # Température Huile
                np.random.gamma(1.5, 0.5, n_samples),     # Vibrations Moteur
                np.random.uniform(30, 40, n_samples)      # Pression Carburant
            ])

            y = np.where(
                (X[:, 0] > 90) | (X[:, 1] > 1.2) | (X[:, 2] < 32) |
                (np.random.rand(n_samples) < 0.1), 1, 0
            )
    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
    
            # Récupération des valeurs utilisateur
            input_values = np.array([float(e.get()) for e in self.entries]).reshape(1, -1)
            prediction = model.predict(input_values)[0]

            # Détermination de la cause si maintenance
            if prediction == 0:
                self.result_label.config(text="🟢 OK", fg="green")
            else:
                temp, vib, press = input_values[0]
                causes = []
                if temp > 90:
                    causes.append("Température huile élevée")
                if vib > 1.2:
                    causes.append("Vibrations moteur excessives")
                if press < 32:
                    causes.append("Pression carburant faible")
                if not causes:
                    causes.append("Défaut aléatoire détecté")
                cause_text = ", ".join(causes)
                self.result_label.config(
                    text=f"🔧 Maintenance requise\n⚠️ Cause : {cause_text}",
                    fg="red"
                )
    
            # Graphe d'importance des variables
            fig, ax = plt.subplots(figsize=(5, 3))
            importances = model.feature_importances_
            features = ["Température Huile", "Vibrations Moteur", "Pression Carburant"]
            ax.barh(features, importances, color='skyblue')
            ax.set_title("Importance des paramètres")
            ax.set_xlabel("Score d'importance")
            ax.set_xlim(0, max(importances) * 1.1)
    
            # Intégrer le graphique dans Tkinter
            self.rf_canvas = FigureCanvasTkAgg(fig, master=self.rf_result_frame)
            self.rf_canvas.draw()
            self.rf_canvas.get_tk_widget().pack(pady=10)

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur dans la prédiction :\n{str(e)}")
    

    def show_arima_input(self):
        """ARIMA avec labels descriptifs"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(main_frame,
                text="TimeSeries ARIMA - Prévision trafic aérien",
                font=('Helvetica', 16, 'bold'),
                bg=self.bg_color).pack(pady=10)
        
        # Frame pour les inputs
        input_frame = tk.Frame(main_frame, bg=self.bg_color)
        input_frame.pack(pady=20)
        
        # Input 1: Mois
        tk.Label(input_frame, text="Mois (séparés par virgules):", 
                bg=self.bg_color).grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.arima_x = tk.Entry(input_frame, width=40)
        self.arima_x.grid(row=0, column=1, padx=5, pady=5)
        self.arima_x.insert(0, "1,2,3,4,5,6,7,8,9,10,11,12")
        
        # Input 2: Nombre de vols
        tk.Label(input_frame, text="Nombre de vols (séparés par virgules):", 
                bg=self.bg_color).grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.arima_y = tk.Entry(input_frame, width=40)
        self.arima_y.grid(row=1, column=1, padx=5, pady=5)
        self.arima_y.insert(0, "100,120,130,110,105,115,125,135,140,130,120,110")
        
        # Input 3: Titre du graphique
        tk.Label(input_frame, text="Titre du graphique:", 
                bg=self.bg_color).grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.arima_title = tk.Entry(input_frame, width=40)
        self.arima_title.grid(row=2, column=1, padx=5, pady=5)
        self.arima_title.insert(0, "Evolution du trafic aérien mensuel")
        
        # Boutons
        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame,
                 text="Générer",
                 command=self.generate_arima,
                 bg=self.button_color,
                 fg='white',
                 font=('Helvetica', 12)).pack(side='left', padx=10)
        
        tk.Button(button_frame,
                 text="Retour",
                 command=self.show_algorithms,
                 bg="#AAAAAA",
                 fg='white',
                 font=('Helvetica', 12)).pack(side='right', padx=10)
        
        # Frame pour le graphique
        self.arima_graph_frame = tk.Frame(main_frame, bg=self.bg_color)
        self.arima_graph_frame.pack(fill='both', expand=True)
    
    def generate_arima(self):
        """Génère la visualisation ARIMA"""
        try:
            X = [x.strip() for x in self.arima_x.get().split(',')]
            Y = [float(y.strip()) for y in self.arima_y.get().split(',')]
            
            if len(X) != len(Y):
                messagebox.showerror("Erreur", "Les listes doivent avoir la même longueur")
                return
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(X, Y, marker='o')
            ax.set_xlabel("Mois")
            ax.set_ylabel("Nombre de vols")
            ax.set_title(self.arima_title.get())
            ax.grid(True)
            
            for widget in self.arima_graph_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.arima_graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
            
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides séparés par des virgules")

    def show_cross_validation(self):
        """Validation croisée comparant Random Forest et Régression Linéaire"""
        self.clear_window()
        
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Titre et description
        tk.Label(main_frame,
                text="Validation Croisée: RF vs Régression Linéaire",
                font=('Helvetica', 16, 'bold'),
                bg=self.bg_color).pack(pady=10)
        
        description = ("Comparaison des performances par validation croisée entre:\n"
                      "- Random Forest (classifieur)\n"
                      "- Régression Linéaire\n\n"
                      "Métriques: MSE (Mean Squared Error) pour la régression")
        tk.Label(main_frame,
                text=description,
                font=('Helvetica', 12),
                bg=self.bg_color).pack(pady=10)
    
        # Paramètres de validation croisée
        param_frame = tk.Frame(main_frame, bg=self.bg_color)
        param_frame.pack(pady=15)
        
        tk.Label(param_frame, 
                text="Nombre de folds (K):",
                bg=self.bg_color).grid(row=0, column=0, padx=5, pady=5)
        self.k_folds = ttk.Spinbox(param_frame, from_=3, to=20, width=5)
        self.k_folds.grid(row=0, column=1, padx=5, pady=5)
        self.k_folds.set(5)
        
        # Bouton d'exécution
        tk.Button(main_frame,
                 text="Exécuter la validation croisée",
                 command=self.run_regression_cv,
                 bg=self.button_color,
                 fg='white',
                 font=('Helvetica', 12)).pack(pady=15)
        # Bouton Retour
        tk.Button(main_frame,
                 text="Retour",
                 command=self.show_algorithms,
                 bg="#AAAAAA",
                 fg='white',
                 font=('Helvetica', 12),
                 width=10).pack(pady=20)
        
        # Zone de résultats
        self.cv_results_frame = tk.Frame(main_frame, bg=self.bg_color)
        self.cv_results_frame.pack(fill='both', expand=True)
    
    def run_regression_cv(self):
        """Exécute la validation croisée pour la régression"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import make_scorer, mean_squared_error
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np
            import pandas as pd
        
            # Nettoyer les résultats précédents
            for widget in self.cv_results_frame.winfo_children():
                widget.destroy()
                
            # Paramètres
            # 1. Chargement des données
            X, y = self.load_regression_data()
            
            # Optionnel : Affichage debug (à retirer en production)
            print("\nAperçu des données chargées:")
            print(pd.DataFrame(X).head())  # Si X est un numpy array, conversion en DataFrame
            print(f"\nDimensions des données - Features: {X.shape}, Target: {y.shape}")
            
            # 2. Paramètres de validation croisée
            k = int(self.k_folds.get())
            
            # 3. Initialisation des modèles 
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100),
                "Régression Linéaire": LinearRegression()
            }

            
            # Métrique MSE
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
            
            # Calcul des scores
            results = {}
            for name, model in models.items():
                cv_scores = -cross_val_score(model, X, y, cv=k, scoring=scorer)
                results[name] = {
                    'scores': cv_scores,
                    'mean': np.mean(cv_scores),
                    'std': np.std(cv_scores)
                }
        
            # Affichage des résultats textuels
            result_text = tk.Text(self.cv_results_frame, height=8, width=80, wrap=tk.WORD)
            result_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            
            result_text.insert(tk.END, "Résultats de validation croisée (MSE):\n\n")
            for name, res in results.items():
                result_text.insert(tk.END, 
                                 f"• {name}:\n"
                                 f"  - MSE moyen: {res['mean']:.3f}\n"
                                 f"  - Écart-type: {res['std']:.3f}\n"
                                 f"  - Scores par fold: {np.round(res['scores'], 3)}\n\n")
            
            # Graphique comparatif
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
            # Boxplot des performances
            ax1.boxplot([results['Random Forest']['scores'], 
                        results['Régression Linéaire']['scores']], 
                       labels=list(models.keys()))
            ax1.set_title("Comparaison des MSE")
            ax1.set_ylabel("Mean Squared Error")
            ax1.grid(True)
            
            # Barplot des moyennes
            means = [results[name]['mean'] for name in models]
            stds = [results[name]['std'] for name in models]
            x_pos = np.arange(len(models))
            
            ax2.bar(x_pos, means, yerr=stds, align='center', alpha=0.7)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(list(models.keys()))
            ax2.set_title("MSE moyen ± écart-type")
            ax2.set_ylabel("MSE")
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Intégration du graphique dans Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.cv_results_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Erreur", f"Erreur lors de la validation croisée:\n{str(e)}")
      
    def load_regression_data(self):
        """Charge des données aéronautiques pour la prédiction de maintenance"""
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)  # Pour la reproductibilité
        
        # Création de données aéronautiques synthétiques réalistes
        n_samples = 200  # Nombre de vols/avions
    
        data = pd.DataFrame({
            # Paramètres avion
            'heures_moteur': np.random.uniform(0, 5000, n_samples),
            'nb_cycles_atterrissage': np.random.poisson(150, n_samples),
            'pression_carburant': np.random.normal(35, 2, n_samples),
            'vibration_moteur': np.random.gamma(1.5, 0.5, n_samples),
            'température_huile': np.random.normal(85, 5, n_samples),
            
            # Conditions de vol
            'altitude_moyenne': np.random.uniform(30000, 40000, n_samples),
            'température_extérieure': np.random.uniform(-40, -20, n_samples),
            
            # Target: Intervalle jusqu'à la prochaine maintenance (en jours)
            'jours_maintenance': np.nan  # À calculer
        })
        
        # Relation synthétique pour la target (formule simplifiée)
        data['jours_maintenance'] = (
            300 - 0.02 * data['heures_moteur'] 
            - 0.5 * data['nb_cycles_atterrissage'] 
            + 2 * data['pression_carburant'] 
            - 10 * data['vibration_moteur'] 
            + np.random.normal(0, 15, n_samples)
        ).clip(1, 365)  # Borne entre 1 et 365 jours
        
        # Séparation features/target
        X = data.drop('jours_maintenance', axis=1)
        y = data['jours_maintenance'].values
        
        return X, y    
            
    def clear_window(self):
        """Efface tous les widgets de la fenêtre"""
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = IAInterface(root)
    root.mainloop()
    
    
   