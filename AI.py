"""
Basic Image Classifier
A comprehensive desktop application for automated image classification using deep learning

Author: Priyanshu Chaurasiya
Registration: 25BCE10136
Subject: Fundamentals in AI/ML
Date: March 27, 2026
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import threading


class ImageClassifierApp:
    """Main application class for the Image Classifier"""
    
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Basic Image Classifier - Aditya Singh (25BCE10157)")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Model variables
        self.model = None
        self.history = None
        self.class_names = []
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.current_image_path = None
        
        # Configuration variables
        self.train_dir = tk.StringVar()
        self.test_dir = tk.StringVar()
        self.epochs_var = tk.IntVar(value=10)
        self.batch_size_var = tk.IntVar(value=32)
        self.img_size_var = tk.IntVar(value=128)
        self.model_type_var = tk.StringVar(value="transfer")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Create notebook (tabbed interface)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#2c3e50')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Arial', 10, 'bold'))
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.training_tab = tk.Frame(self.notebook, bg='#ecf0f1')
        self.prediction_tab = tk.Frame(self.notebook, bg='#ecf0f1')
        self.visualization_tab = tk.Frame(self.notebook, bg='#ecf0f1')
        
        self.notebook.add(self.training_tab, text='📚 Model Training')
        self.notebook.add(self.prediction_tab, text='🔍 Prediction')
        self.notebook.add(self.visualization_tab, text='📊 Visualization')
        
        # Setup individual tabs
        self.setup_training_tab()
        self.setup_prediction_tab()
        self.setup_visualization_tab()
        
    def setup_training_tab(self):
        """Setup the training tab interface"""
        # Title
        title_label = tk.Label(
            self.training_tab, 
            text="Train Your Image Classification Model",
            font=('Arial', 18, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Dataset selection frame
        dataset_frame = tk.LabelFrame(
            self.training_tab,
            text="📁 Dataset Selection",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#34495e',
            padx=20,
            pady=20
        )
        dataset_frame.pack(fill='x', padx=20, pady=10)
        
        # Training directory
        tk.Label(dataset_frame, text="Training Directory:", bg='#ecf0f1', font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=5)
        tk.Entry(dataset_frame, textvariable=self.train_dir, width=50, font=('Arial', 10)).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(
            dataset_frame, 
            text="Browse", 
            command=self.browse_train_dir,
            bg='#3498db',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2'
        ).grid(row=0, column=2, pady=5)
        
        # Test directory
        tk.Label(dataset_frame, text="Test Directory:", bg='#ecf0f1', font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=5)
        tk.Entry(dataset_frame, textvariable=self.test_dir, width=50, font=('Arial', 10)).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(
            dataset_frame, 
            text="Browse", 
            command=self.browse_test_dir,
            bg='#3498db',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2'
        ).grid(row=1, column=2, pady=5)
        
        # Load dataset button
        tk.Button(
            dataset_frame,
            text="📂 Load Dataset",
            command=self.load_dataset,
            bg='#2ecc71',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            padx=20,
            pady=5
        ).grid(row=2, column=1, pady=15)
        
        # Model configuration frame
        config_frame = tk.LabelFrame(
            self.training_tab,
            text="⚙️ Model Configuration",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#34495e',
            padx=20,
            pady=20
        )
        config_frame.pack(fill='x', padx=20, pady=10)
        
        # Parameters
        tk.Label(config_frame, text="Epochs:", bg='#ecf0f1', font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=5)
        tk.Spinbox(config_frame, from_=1, to=100, textvariable=self.epochs_var, width=15, font=('Arial', 10)).grid(row=0, column=1, padx=10, pady=5)
        
        tk.Label(config_frame, text="Batch Size:", bg='#ecf0f1', font=('Arial', 10)).grid(row=0, column=2, sticky='w', pady=5, padx=20)
        tk.Spinbox(config_frame, from_=8, to=128, textvariable=self.batch_size_var, width=15, font=('Arial', 10)).grid(row=0, column=3, padx=10, pady=5)
        
        tk.Label(config_frame, text="Image Size:", bg='#ecf0f1', font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=5)
        tk.Spinbox(config_frame, from_=32, to=512, textvariable=self.img_size_var, width=15, font=('Arial', 10)).grid(row=1, column=1, padx=10, pady=5)
        
        # Model type selection
        tk.Label(config_frame, text="Model Type:", bg='#ecf0f1', font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=10)
        tk.Radiobutton(
            config_frame, 
            text="Custom CNN", 
            variable=self.model_type_var, 
            value="cnn",
            bg='#ecf0f1',
            font=('Arial', 10)
        ).grid(row=2, column=1, sticky='w', pady=10)
        tk.Radiobutton(
            config_frame, 
            text="Transfer Learning (MobileNetV2)", 
            variable=self.model_type_var, 
            value="transfer",
            bg='#ecf0f1',
            font=('Arial', 10)
        ).grid(row=2, column=2, columnspan=2, sticky='w', pady=10)
        
        # Action buttons frame
        button_frame = tk.Frame(self.training_tab, bg='#ecf0f1')
        button_frame.pack(pady=20)
        
        tk.Button(
            button_frame,
            text="🚀 Train Model",
            command=self.train_model_thread,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side='left', padx=10)
        
        tk.Button(
            button_frame,
            text="📊 Evaluate Model",
            command=self.evaluate_model,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side='left', padx=10)
        
        tk.Button(
            button_frame,
            text="💾 Save Model",
            command=self.save_model,
            bg='#16a085',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side='left', padx=10)
        
        tk.Button(
            button_frame,
            text="📥 Load Model",
            command=self.load_model,
            bg='#f39c12',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side='left', padx=10)
        
        # Console output
        console_frame = tk.LabelFrame(
            self.training_tab,
            text="📝 Training Console",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#34495e',
            padx=10,
            pady=10
        )
        console_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.console = scrolledtext.ScrolledText(
            console_frame,
            height=10,
            font=('Courier', 9),
            bg='#2c3e50',
            fg='#2ecc71',
            insertbackground='white'
        )
        self.console.pack(fill='both', expand=True)
        
    def setup_prediction_tab(self):
        """Setup the prediction tab interface"""
        # Title
        title_label = tk.Label(
            self.prediction_tab,
            text="Classify Your Images",
            font=('Arial', 18, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Image display frame
        image_frame = tk.LabelFrame(
            self.prediction_tab,
            text="🖼️ Selected Image",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#34495e',
            padx=20,
            pady=20
        )
        image_frame.pack(pady=20)
        
        self.image_label = tk.Label(image_frame, bg='#ecf0f1', text="No image loaded", font=('Arial', 10))
        self.image_label.pack()
        
        # Buttons frame
        button_frame = tk.Frame(self.prediction_tab, bg='#ecf0f1')
        button_frame.pack(pady=20)
        
        tk.Button(
            button_frame,
            text="📂 Load Image",
            command=self.load_image,
            bg='#3498db',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side='left', padx=10)
        
        tk.Button(
            button_frame,
            text="🔮 Predict",
            command=self.predict_image,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            padx=20,
            pady=10
        ).pack(side='left', padx=10)
        
        # Results frame
        results_frame = tk.LabelFrame(
            self.prediction_tab,
            text="📊 Prediction Results",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1',
            fg='#34495e',
            padx=20,
            pady=20
        )
        results_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            font=('Arial', 11),
            bg='white',
            fg='#2c3e50'
        )
        self.results_text.pack(fill='both', expand=True)
        
    def setup_visualization_tab(self):
        """Setup the visualization tab interface"""
        # Title
        title_label = tk.Label(
            self.visualization_tab,
            text="Model Performance Visualization",
            font=('Arial', 18, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Buttons frame
        button_frame = tk.Frame(self.visualization_tab, bg='#ecf0f1')
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text="📈 Show Training History",
            command=self.plot_training_history,
            bg='#3498db',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            padx=15,
            pady=8
        ).pack(side='left', padx=10)
        
        tk.Button(
            button_frame,
            text="🎯 Show Confusion Matrix",
            command=self.plot_confusion_matrix,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 11, 'bold'),
            cursor='hand2',
            padx=15,
            pady=8
        ).pack(side='left', padx=10)
        
        # Visualization canvas frame
        self.viz_frame = tk.Frame(self.visualization_tab, bg='#ecf0f1')
        self.viz_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
    # ==================== DATASET METHODS ====================
    
    def browse_train_dir(self):
        """Browse for training directory"""
        directory = filedialog.askdirectory(title="Select Training Directory")
        if directory:
            self.train_dir.set(directory)
            
    def browse_test_dir(self):
        """Browse for test directory"""
        directory = filedialog.askdirectory(title="Select Test Directory")
        if directory:
            self.test_dir.set(directory)
            
    def load_dataset(self):
        """Load and preprocess the dataset"""
        try:
            train_path = self.train_dir.get()
            test_path = self.test_dir.get()
            
            if not train_path or not test_path:
                messagebox.showerror("Error", "Please select both training and test directories!")
                return
            
            self.log_console("Loading dataset...\n")
            img_size = self.img_size_var.get()
            
            # Load training data
            self.log_console(f"Loading training images from: {train_path}\n")
            self.train_images, self.train_labels, self.class_names = self.load_images_from_directory(train_path, img_size)
            
            # Load test data
            self.log_console(f"Loading test images from: {test_path}\n")
            self.test_images, self.test_labels, _ = self.load_images_from_directory(test_path, img_size)
            
            self.log_console(f"\n✅ Dataset loaded successfully!\n")
            self.log_console(f"Training samples: {len(self.train_images)}\n")
            self.log_console(f"Test samples: {len(self.test_images)}\n")
            self.log_console(f"Number of classes: {len(self.class_names)}\n")
            self.log_console(f"Classes: {', '.join(self.class_names)}\n")
            
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            
        except Exception as e:
            self.log_console(f"❌ Error loading dataset: {str(e)}\n")
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            
    def load_images_from_directory(self, directory, img_size):
        """Load images from directory structure"""
        images = []
        labels = []
        class_names = sorted([d for d in os.listdir(directory) 
                             if os.path.isdir(os.path.join(directory, d))])
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(directory, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            for img_file in image_files:
                try:
                    img_path = os.path.join(class_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (img_size, img_size))
                        img = img.astype('float32') / 255.0
                        images.append(img)
                        labels.append(class_idx)
                except Exception as e:
                    self.log_console(f"Warning: Could not load {img_file}: {str(e)}\n")
                    
        return np.array(images), np.array(labels), class_names
    
    # ==================== MODEL METHODS ====================
    
    def create_cnn_model(self, img_size, num_classes):
        """Create a custom CNN model"""
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Classification head
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self, img_size, num_classes):
        """Create a transfer learning model using MobileNetV2"""
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model_thread(self):
        """Start training in a separate thread"""
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def train_model(self):
        """Train the classification model"""
        try:
            if self.train_images is None:
                messagebox.showerror("Error", "Please load dataset first!")
                return
            
            self.log_console("\n" + "="*60 + "\n")
            self.log_console("🚀 Starting model training...\n")
            self.log_console("="*60 + "\n\n")
            
            img_size = self.img_size_var.get()
            num_classes = len(self.class_names)
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            model_type = self.model_type_var.get()
            
            # Create model
            self.log_console(f"Creating {model_type.upper()} model...\n")
            if model_type == "cnn":
                self.model = self.create_cnn_model(img_size, num_classes)
            else:
                self.model = self.create_transfer_learning_model(img_size, num_classes)
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.log_console(f"\nModel Summary:\n")
            self.log_console(f"Total parameters: {self.model.count_params():,}\n")
            self.log_console(f"Image size: {img_size}x{img_size}\n")
            self.log_console(f"Classes: {num_classes}\n")
            self.log_console(f"Epochs: {epochs}\n")
            self.log_console(f"Batch size: {batch_size}\n\n")
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.2,
                    patience=3,
                    verbose=1
                )
            ]
            
            # Train model
            self.log_console("Training started...\n\n")
            self.history = self.model.fit(
                self.train_images,
                self.train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Log training results
            for epoch in range(len(self.history.history['loss'])):
                train_loss = self.history.history['loss'][epoch]
                train_acc = self.history.history['accuracy'][epoch]
                val_loss = self.history.history['val_loss'][epoch]
                val_acc = self.history.history['val_accuracy'][epoch]
                
                self.log_console(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                    f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}\n"
                )
            
            self.log_console("\n" + "="*60 + "\n")
            self.log_console("✅ Training completed successfully!\n")
            self.log_console("="*60 + "\n")
            
            messagebox.showinfo("Success", "Model training completed!")
            
        except Exception as e:
            self.log_console(f"\n❌ Error during training: {str(e)}\n")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        try:
            if self.model is None:
                messagebox.showerror("Error", "Please train or load a model first!")
                return
            
            if self.test_images is None:
                messagebox.showerror("Error", "Please load test dataset first!")
                return
            
            self.log_console("\n" + "="*60 + "\n")
            self.log_console("📊 Evaluating model...\n")
            self.log_console("="*60 + "\n\n")
            
            # Evaluate on test set
            test_loss, test_accuracy = self.model.evaluate(
                self.test_images,
                self.test_labels,
                verbose=0
            )
            
            # Generate predictions
            predictions = self.model.predict(self.test_images, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Classification report
            report = classification_report(
                self.test_labels,
                predicted_classes,
                target_names=self.class_names,
                digits=4
            )
            
            self.log_console(f"Test Loss: {test_loss:.4f}\n")
            self.log_console(f"Test Accuracy: {test_accuracy:.4f}\n\n")
            self.log_console("Classification Report:\n")
            self.log_console(report + "\n")
            
            messagebox.showinfo("Evaluation Complete", 
                              f"Test Accuracy: {test_accuracy*100:.2f}%\n"
                              f"Test Loss: {test_loss:.4f}")
            
        except Exception as e:
            self.log_console(f"\n❌ Error during evaluation: {str(e)}\n")
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
    
    def save_model(self):
        """Save the trained model"""
        try:
            if self.model is None:
                messagebox.showerror("Error", "No model to save!")
                return
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".keras",
                filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")]
            )
            
            if file_path:
                self.model.save(file_path)
                self.log_console(f"\n✅ Model saved to: {file_path}\n")
                messagebox.showinfo("Success", "Model saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load a trained model"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Keras Model", "*.keras"), ("All Files", "*.*")]
            )
            
            if file_path:
                self.model = keras.models.load_model(file_path)
                self.log_console(f"\n✅ Model loaded from: {file_path}\n")
                messagebox.showinfo("Success", "Model loaded successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    # ==================== PREDICTION METHODS ====================
    
    def load_image(self):
        """Load an image for prediction"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
            )
            
            if file_path:
                self.current_image_path = file_path
                
                # Display image
                img = Image.open(file_path)
                img = img.resize((200, 200))
                photo = ImageTk.PhotoImage(img)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Image loaded: {os.path.basename(file_path)}\n")
                self.results_text.insert(tk.END, "Click 'Predict' to classify the image.\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def predict_image(self):
        """Predict the class of the loaded image"""
        try:
            if self.model is None:
                messagebox.showerror("Error", "Please train or load a model first!")
                return
            
            if self.current_image_path is None:
                messagebox.showerror("Error", "Please load an image first!")
                return
            
            # Preprocess image
            img_size = self.img_size_var.get()
            img = cv2.imread(self.current_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32') / 255.0
            img_array = np.expand_dims(img, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "="*50 + "\n")
            self.results_text.insert(tk.END, "PREDICTION RESULTS\n")
            self.results_text.insert(tk.END, "="*50 + "\n\n")
            
            self.results_text.insert(tk.END, f"🎯 Predicted Class: {self.class_names[predicted_class]}\n", "bold")
            self.results_text.insert(tk.END, f"📊 Confidence: {confidence*100:.2f}%\n\n")
            
            self.results_text.insert(tk.END, "All Class Probabilities:\n")
            self.results_text.insert(tk.END, "-"*50 + "\n")
            
            # Sort predictions by probability
            sorted_indices = np.argsort(predictions[0])[::-1]
            for idx in sorted_indices:
                class_name = self.class_names[idx]
