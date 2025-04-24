# Handwriting Personality Analyzer
# Enhanced version with UI and DSBDA improvements

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import datetime
import json
import seaborn as sns

class HandwritingAnalyzer:
    """
    Core class for analyzing handwriting samples and extracting features
    """
    def __init__(self):
        self.standard_size = (1240, 1754)  # A4 in pixels at ~150 DPI
        
    def preprocess_image(self, image):
        """
        Convert to grayscale, blur, and binarize the image.
        Adaptive thresholding is used for robust thresholding under different lighting.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Adaptive thresholding for variable lighting conditions
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 
                                      15, 10)
        return thresh

    def get_written_area(self, thresh, size=(1240, 1754)):
        """
        Find contours and select the largest bounding rectangle which presumably
        corresponds to the handwritten area. 
        Use morphology to merge nearby components if needed.
        """
        # Use dilation to merge individual handwriting strokes if necessary.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours on the dilated image.
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours based on area and position
        img_area = size[0] * size[1]
        W, H = size
        valid_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            if not (0.0002 * img_area < area < 0.04 * img_area):
                continue

            too_close_left = x < 0.02 * W
            too_close_bottom = (y + h) > 0.98 * H
            aspect_ratio = h / (w + 1e-5)

            # Reject small blobs on the border or weird tall vertical ones
            if (too_close_left or too_close_bottom) and area < 800:
                continue
            if too_close_left and aspect_ratio > 5 and area > 100:
                continue  # likely left-side shading or scan artifact
            
            valid_boxes.append((x, y, w, h))

        if not valid_boxes:
            return None

        # Find extremes of all boxes
        x_vals = [x for x, y, w, h in valid_boxes]
        y_vals = [y for x, y, w, h in valid_boxes]
        x2_vals = [x + w for x, y, w, h in valid_boxes]
        y2_vals = [y + h for x, y, w, h in valid_boxes]

        # Unified bounding box
        x_min = min(x_vals)
        y_min = min(y_vals)
        x_max = max(x2_vals)
        y_max = max(y2_vals)

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return bbox

    def analyze_margins(self, image, bounding_box, margin_threshold=50):
        """
        Calculates margin distances from the bounding box to page edges.
        Compares each margin with a threshold (in pixels) to determine if the margin is 'good'.
        Returns margins analysis and actual margin measurements.
        """
        img_h, img_w = image.shape[:2]
        x, y, w, h = bounding_box
        
        # Calculate distances
        left_margin = x
        top_margin = y
        right_margin = img_w - (x + w)
        bottom_margin = img_h - (y + h)
        
        # Evaluate if each margin is above the threshold
        left_good = left_margin >= margin_threshold
        right_good = right_margin >= margin_threshold
        top_good = top_margin >= margin_threshold
        bottom_good = bottom_margin >= margin_threshold
        
        # Calculate margin ratios (normalized metrics)
        total_width = img_w
        total_height = img_h
        left_ratio = left_margin / total_width
        right_ratio = right_margin / total_width
        top_ratio = top_margin / total_height
        bottom_ratio = bottom_margin / total_height
        
        margin_bools = [left_good, right_good, top_good, bottom_good]
        margin_values = [left_margin, top_margin, right_margin, bottom_margin]
        margin_ratios = [left_ratio, right_ratio, top_ratio, bottom_ratio]
        
        return margin_bools, margin_values, margin_ratios

    def analyze_line_orientation(self, thresh, debug=False):
        """
        Use probabilistic Hough Line Transform to detect lines in the threshold image.
        Analyze the angles of the detected lines to determine if lines are straight, sloped, or curved.
        Returns line analysis booleans and detailed metrics.
        """
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        # Using HoughLinesP to get line segments
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                              minLineLength=thresh.shape[1]//4, maxLineGap=20)
        
        if lines is None or len(lines) < 3:
            # No lines or too few lines detected
            return [False, False, False], {"mean_angle": 0, "std_angle": 0, "num_lines": 0}
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Avoid division by zero
            if x2 - x1 == 0:
                angle = 90.0
            else:
                angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
            angles.append(angle)
            
        angles = np.array(angles)
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        # Define thresholds (these may need tuning)
        angle_straight_threshold = 5.0  # if mean angle is within 5° of horizontal, count as straight
        std_angle_threshold = 10.0      # low variance implies consistency

        is_line_straight = (abs(mean_angle) <= angle_straight_threshold and std_angle <= std_angle_threshold)
        is_line_sloped = (abs(mean_angle) > angle_straight_threshold and std_angle <= std_angle_threshold)
        is_line_curved = std_angle > std_angle_threshold

        line_bools = [is_line_straight, is_line_sloped, is_line_curved]
        line_metrics = {
            "mean_angle": mean_angle, 
            "std_angle": std_angle, 
            "num_lines": len(lines)
        }
        
        return line_bools, line_metrics

    def extract_word_spacing(self, thresh):
        """
        Extract word spacing features from the binary image.
        Returns metrics about average spacing between words.
        """
        # Use horizontal projection to analyze word spacing
        h_proj = np.sum(thresh, axis=0)
        
        # Find transitions (potential word boundaries)
        transitions = []
        state = 0  # 0 = white space, 1 = text
        for i in range(1, len(h_proj)):
            if h_proj[i-1] == 0 and h_proj[i] > 0:
                state = 1  # transition to text
            elif h_proj[i-1] > 0 and h_proj[i] == 0:
                state = 0  # transition to space
                transitions.append(i)
        
        if len(transitions) < 2:
            return {
                "avg_spacing": 0,
                "std_spacing": 0,
                "spacing_density": 0
            }
            
        # Calculate spaces between transitions
        spaces = [transitions[i] - transitions[i-1] for i in range(1, len(transitions))]
        
        if not spaces:
            return {
                "avg_spacing": 0,
                "std_spacing": 0,
                "spacing_density": 0
            }
            
        avg_spacing = np.mean(spaces)
        std_spacing = np.std(spaces)
        spacing_density = len(spaces) / thresh.shape[1]  # Normalized by image width
        
        return {
            "avg_spacing": avg_spacing,
            "std_spacing": std_spacing,
            "spacing_density": spacing_density
        }

    def extract_additional_features(self, thresh, bbox):
        """
        Extract additional features from the handwriting:
        - Text density
        - Line height variation
        - Horizontal alignment consistency
        """
        if bbox is None:
            return {
                "text_density": 0,
                "line_height_variation": 0,
                "horizontal_alignment": 0
            }
            
        x, y, w, h = bbox
        roi = thresh[y:y+h, x:x+w]
        
        # Text density (ratio of ink pixels to total area)
        text_pixels = np.sum(roi > 0)
        total_pixels = roi.size
        text_density = text_pixels / total_pixels
        
        # Line height analysis
        h_proj = np.sum(roi, axis=1)
        lines = []
        in_line = False
        start = 0
        
        for i in range(len(h_proj)):
            if not in_line and h_proj[i] > 0:
                in_line = True
                start = i
            elif in_line and h_proj[i] == 0:
                in_line = False
                lines.append((start, i))
        
        if lines:
            line_heights = [end - start for start, end in lines]
            line_height_variation = np.std(line_heights) / (np.mean(line_heights) + 1e-6)
        else:
            line_height_variation = 0
        
        # Horizontal alignment consistency
        v_proj = np.sum(roi, axis=0)
        left_edges = []
        in_col = False
        
        for i in range(len(v_proj)):
            if not in_col and v_proj[i] > 0:
                in_col = True
                left_edges.append(i)
            elif in_col and v_proj[i] == 0:
                in_col = False
        
        if left_edges:
            horizontal_alignment = np.std(left_edges) / w  # Normalized by width
        else:
            horizontal_alignment = 0
            
        return {
            "text_density": text_density,
            "line_height_variation": line_height_variation,
            "horizontal_alignment": horizontal_alignment
        }

    def process_image(self, image_path, margin_threshold=50):
        """
        Process a handwriting image and extract all features.
        Returns a dictionary of features and processed image data.
        """
        try:
            # Read and resize the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            image = cv2.resize(image, self.standard_size, interpolation=cv2.INTER_AREA)
            
            # Preprocess the image
            thresh = self.preprocess_image(image)
            
            # Get bounding box for written area
            bbox = self.get_written_area(thresh)
            
            if bbox is None:
                return {
                    "success": False,
                    "error": "No writing detected in the image."
                }
                
            # Extract features
            margin_bools, margin_values, margin_ratios = self.analyze_margins(
                image, bbox, margin_threshold)
            
            line_bools, line_metrics = self.analyze_line_orientation(thresh)
            
            spacing_metrics = self.extract_word_spacing(thresh)
            
            additional_features = self.extract_additional_features(thresh, bbox)
            
            # Combine all features
            features = {
                "left_margin_good": int(margin_bools[0]),
                "right_margin_good": int(margin_bools[1]),
                "top_margin_good": int(margin_bools[2]),
                "bottom_margin_good": int(margin_bools[3]),
                "is_line_straight": int(line_bools[0]),
                "is_line_sloped": int(line_bools[1]),
                "is_line_curved": int(line_bools[2]),
                "left_margin": margin_values[0],
                "top_margin": margin_values[1],
                "right_margin": margin_values[2],
                "bottom_margin": margin_values[3],
                "left_margin_ratio": margin_ratios[0],
                "right_margin_ratio": margin_ratios[1],
                "top_margin_ratio": margin_ratios[2],
                "bottom_margin_ratio": margin_ratios[3],
                "mean_line_angle": line_metrics["mean_angle"],
                "std_line_angle": line_metrics["std_angle"],
                "num_lines": line_metrics["num_lines"],
                "avg_word_spacing": spacing_metrics["avg_spacing"],
                "std_word_spacing": spacing_metrics["std_spacing"],
                "spacing_density": spacing_metrics["spacing_density"],
                "text_density": additional_features["text_density"],
                "line_height_variation": additional_features["line_height_variation"],
                "horizontal_alignment": additional_features["horizontal_alignment"]
            }
            
            # Prepare results
            result = {
                "success": True,
                "features": features,
                "bbox": bbox,
                "image": image,
                "thresh": thresh
            }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class PersonalityPredictor:
    """
    ML model for predicting personality traits from handwriting features
    """
    def __init__(self):
        self.model = None
        self.personality_traits = {
            'family_attachment': {
                True: "Strong attachment to family, culture, and tradition. Likely to stay in the same job for a long time.",
                False: "Socially conscious but with weaker emotional bonds to family. More independent."
            },
            'risk_taking': {
                True: "Fear of failure with follower mentality. May experience stagnation over time.",
                False: "Bold risk-taker with impulsive approach to goals. Adventurous and opportunity-seeking."
            },
            'energy': {
                True: "Positive energy and optimistic outlook.",
                False: "More reserved energy and pragmatic attitude."
            },
            'planning': {
                True: "Highly productive with effective resource utilization. Strong vision for the future.",
                False: "Acts without hesitation but tends to over-plan every moment."
            },
            'personality_type': {
                'introvert': "Introvert: Derives energy from solitude and internal reflection. Prefers deeper one-on-one interactions.",
                'extrovert': "Extrovert: Energized by social interactions. Outgoing and comfortable in group settings.",
                'ambivert': "Ambivert: Balanced between introversion and extroversion. Adaptable to different social contexts."
            }
        }
    
    def load_model(self, model_path):
        """Load a trained ML model"""
        try:
            self.model = joblib.load(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path):
        """Save the current ML model"""
        if self.model:
            joblib.dump(self.model, model_path)
            return True
        return False
    
    def train_model(self, features_df, labels):
        """Train a new model with the provided features and labels"""
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42)
        
        # Initialize and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return {
            "accuracy": accuracy,
            "report": report,
            "model": self.model
        }
    
    def predict_basic_traits(self, features):
        """
        Rule-based prediction of basic personality traits
        This acts as a fallback when no ML model is available
        """
        traits = {}
        
        # Family attachment (based on left margin)
        traits['family_attachment'] = not features.get('left_margin_good', False)
        
        # Risk taking (based on right margin)
        traits['risk_taking'] = features.get('right_margin_good', False)
        
        # Energy levels (based on line slope)
        traits['energy'] = features.get('is_line_sloped', False)
        
        # Planning abilities (based on bottom margin)
        traits['planning'] = features.get('bottom_margin_good', False)
        
        # Basic personality type prediction (simplified)
        if traits['family_attachment'] and not traits['risk_taking']:
            traits['personality_type'] = 'introvert'
        elif not traits['family_attachment'] and not traits['risk_taking']:
            traits['personality_type'] = 'extrovert'
        else:
            traits['personality_type'] = 'ambivert'
            
        return traits
    
    def predict_ml_traits(self, features_dict):
        """
        Use the trained ML model to predict personality traits
        """
        if not self.model:
            return None
        
        # Extract the features expected by the model
        try:
            # Convert dict to DataFrame for prediction
            features_df = pd.DataFrame([features_dict])
            
            # Make prediction
            prediction = self.model.predict(features_df)[0]
            
            # Return prediction
            return prediction
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return None
            
    def get_personality_description(self, traits):
        """Generate textual description of personality based on traits"""
        description = ""
        
        # Add descriptions for each trait
        for trait, value in traits.items():
            if trait == 'personality_type':
                description += self.personality_traits[trait][value] + "\n\n"
            else:
                description += self.personality_traits[trait][value] + "\n\n"
                
        return description.strip()


class DataManager:
    """
    Handle data collection, storage, and retrieval for the system
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.features_file = os.path.join(data_dir, "features.csv")
        self.results_file = os.path.join(data_dir, "analysis_results.json")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Initialize features DataFrame
        if os.path.exists(self.features_file):
            self.features_df = pd.read_csv(self.features_file)
        else:
            self.features_df = pd.DataFrame()
            
    def save_analysis_result(self, image_path, features, traits, timestamp=None):
        """Save analysis results to JSON and CSV"""
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        # Add to features DataFrame
        features_row = features.copy()
        features_row['image_path'] = os.path.basename(image_path)
        features_row['timestamp'] = timestamp
        
        # Add traits
        for trait, value in traits.items():
            features_row[f'trait_{trait}'] = value
            
        # Append to DataFrame
        self.features_df = pd.concat([self.features_df, pd.DataFrame([features_row])], 
                                    ignore_index=True)
        
        # Save to CSV
        self.features_df.to_csv(self.features_file, index=False)
        
        # Save detailed results to JSON
        result = {
            "image_path": image_path,
            "timestamp": timestamp,
            "features": features,
            "traits": traits
        }
        
        # Append to JSON file
        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                try:
                    results = json.load(f)
                except:
                    results = []
        
        results.append(result)
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return True
        
    def get_all_features(self):
        """Return all features as DataFrame"""
        return self.features_df
        
    def get_samples_count(self):
        """Return count of analyzed samples"""
        return len(self.features_df)


class VisualizationManager:
    """
    Generate visualizations for analysis results
    """
    def __init__(self):
        pass
        
    def draw_bounding_box(self, image, bbox):
        """Draw bounding box on image"""
        img_copy = image.copy()
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img_copy
        
    def create_feature_histogram(self, features_df, feature_name):
        """Create histogram for a specific feature"""
        plt.figure(figsize=(8, 4))
        sns.histplot(features_df[feature_name].dropna(), kde=True)
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Count')
        
        # Save to buffer for displaying in UI
        fig = plt.gcf()
        return fig
        
    def create_correlation_heatmap(self, features_df):
        """Create correlation heatmap for numerical features"""
        # Select only numeric columns
        numeric_df = features_df.select_dtypes(include=['float64', 'int64'])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        
        # Save to buffer for displaying in UI
        fig = plt.gcf()
        return fig
        
    def create_trait_distribution_pie(self, features_df, trait_name):
        """Create pie chart for trait distribution"""
        trait_col = f'trait_{trait_name}'
        if trait_col in features_df.columns:
            trait_counts = features_df[trait_col].value_counts()
            
            plt.figure(figsize=(8, 8))
            plt.pie(trait_counts, labels=trait_counts.index, autopct='%1.1f%%', 
                   shadow=True, startangle=90)
            plt.axis('equal')
            plt.title(f'Distribution of {trait_name}')
            
            # Save to buffer for displaying in UI
            fig = plt.gcf()
            return fig
        return None


class HandwritingAnalysisApp:
    """
    Main application class with UI components
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Personality Analyzer")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.analyzer = HandwritingAnalyzer()
        self.predictor = PersonalityPredictor()
        self.data_manager = DataManager()
        self.visualizer = VisualizationManager()
        
        # Try to load pre-trained model
        model_path = os.path.join("models", "personality_model.pkl")
        if os.path.exists(model_path):
            self.predictor.load_model(model_path)
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the application UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.analysis_tab = ttk.Frame(self.notebook)
        self.batch_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.analysis_tab, text="Analysis")
        self.notebook.add(self.batch_tab, text="Batch Processing")
        self.notebook.add(self.data_tab, text="Data Management")
        self.notebook.add(self.stats_tab, text="Statistics")
        
        # Setup individual tabs
        self.setup_analysis_tab()
        self.setup_batch_tab()
        self.setup_data_tab()
        self.setup_stats_tab()
        
    def setup_analysis_tab(self):
        """Set up the Single Image Analysis tab"""
        # Left panel for image and controls
        left_panel = ttk.Frame(self.analysis_tab, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel for results
        right_panel = ttk.Frame(self.analysis_tab, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image selection controls
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(controls_frame, text="Select handwriting image:").pack(side=tk.LEFT)
        
        select_btn = ttk.Button(controls_frame, text="Browse...", 
                               command=self.select_image)
        select_btn.pack(side=tk.LEFT, padx=5)
        
        analyze_btn = ttk.Button(controls_frame, text="Analyze", 
                                command=self.analyze_single_image)
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display
        image_frame = ttk.LabelFrame(left_panel, text="Handwriting Image")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Results display
        results_frame = ttk.LabelFrame(right_panel, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Personality results
        personality_frame = ttk.LabelFrame(right_panel, text="Personality Profile")
        personality_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.personality_text = tk.Text(personality_frame, wrap=tk.WORD, height=15)
        self.personality_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.analysis_tab, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_batch_tab(self):
        """Set up the Batch Processing tab"""
        # Top controls
        controls_frame = ttk.Frame(self.batch_tab, padding="10")
        controls_frame.pack(fill=tk.X)
        
        ttk.Label(controls_frame, text="Select folder with handwriting samples:").pack(side=tk.LEFT)
        
        select_folder_btn = ttk.Button(controls_frame, text="Browse...", 
                                      command=self.select_batch_folder)
        select_folder_btn.pack(side=tk.LEFT, padx=5)
        
        process_btn = ttk.Button(controls_frame, text="Process Batch", 
                               command=self.process_batch)
        process_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress indicator
        progress_frame = ttk.Frame(self.batch_tab, padding="10")
        progress_frame.pack(fill=tk.X)
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        
        self.batch_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, 
                                            length=400, mode='determinate')
        self.batch_progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.batch_status_var = tk.StringVar()
        self.batch_status_var.set("Ready")
        ttk.Label(progress_frame, textvariable=self.batch_status_var).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.batch_tab, text="Batch Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create treeview for batch results
        columns = ("filename", "personality_type", "family_attachment", "risk_taking", "energy", "planning")
        self.batch_results_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
        
        # Define headings
        self.batch_results_tree.heading("filename", text="Filename")
        self.batch_results_tree.heading("personality_type", text="Personality Type")
        self.batch_results_tree.heading("family_attachment", text="Family Attachment")
        self.batch_results_tree.heading("risk_taking", text="Risk Taking")
        self.batch_results_tree.heading("energy", text="Energy")
        self.batch_results_tree.heading("planning", text="Planning")
        
        # Column widths
        self.batch_results_tree.column("filename", width=200)
        self.batch_results_tree.column("personality_type", width=120)
        self.batch_results_tree.column("family_attachment", width=120)
        self.batch_results_tree.column("risk_taking", width=120)
        self.batch_results_tree.column("energy", width=80)
        self.batch_results_tree.column("planning", width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.batch_results_tree.yview)
        self.batch_results_tree.configure(yscroll=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Export button
        export_btn = ttk.Button(self.batch_tab, text="Export Results", command=self.export_batch_results)
        export_btn.pack(pady=10)
        
    def setup_data_tab(self):
        """Set up the Data Management tab"""
        # Top controls
        controls_frame = ttk.Frame(self.data_tab, padding="10")
        controls_frame.pack(fill=tk.X)
        
        ttk.Label(controls_frame, text="Data Management").pack(side=tk.LEFT)
        
        refresh_btn = ttk.Button(controls_frame, text="Refresh Data", 
                                command=self.refresh_data)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        train_model_btn = ttk.Button(controls_frame, text="Train Model", 
                                    command=self.train_new_model)
        train_model_btn.pack(side=tk.LEFT, padx=5)
        
        # Data display
        data_frame = ttk.LabelFrame(self.data_tab, text="Collected Data", padding="10")
        data_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create treeview for data
        self.data_tree = ttk.Treeview(data_frame)
        
        # Configure scrollbars
        y_scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        x_scrollbar = ttk.Scrollbar(data_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscroll=y_scrollbar.set, xscroll=x_scrollbar.set)
        
        # Layout
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Data stats frame
        stats_frame = ttk.LabelFrame(self.data_tab, text="Dataset Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.data_stats_text = tk.Text(stats_frame, height=5, wrap=tk.WORD)
        self.data_stats_text.pack(fill=tk.BOTH)
        
    def setup_stats_tab(self):
        """Set up the Statistics tab"""
        # Controls frame
        controls_frame = ttk.Frame(self.stats_tab, padding="10")
        controls_frame.pack(fill=tk.X)
        
        ttk.Label(controls_frame, text="Select Visualization:").pack(side=tk.LEFT)
        
        # Visualization selector
        self.viz_type_var = tk.StringVar()
        viz_options = [
            "Feature Distributions", 
            "Trait Distributions", 
            "Feature Correlations",
            "Personality Type Distribution"
        ]
        viz_dropdown = ttk.Combobox(controls_frame, textvariable=self.viz_type_var, 
                                   values=viz_options, state="readonly")
        viz_dropdown.pack(side=tk.LEFT, padx=5)
        viz_dropdown.current(0)
        
        viz_dropdown.bind("<<ComboboxSelected>>", self.update_visualization)
        
        # Feature/trait selector (shown conditionally)
        self.feature_var = tk.StringVar()
        self.feature_dropdown = ttk.Combobox(controls_frame, textvariable=self.feature_var, 
                                           state="readonly")
        self.feature_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        generate_btn = ttk.Button(controls_frame, text="Generate", 
                                 command=self.generate_visualization)
        generate_btn.pack(side=tk.LEFT, padx=5)
        
        # Visualization frame
        self.viz_frame = ttk.LabelFrame(self.stats_tab, text="Visualization", padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Canvas for matplotlib figures
        self.viz_canvas_frame = ttk.Frame(self.viz_frame)
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Handwriting Sample",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.status_var.set(f"Selected: {os.path.basename(file_path)}")
            
            # Display the image
            self.display_image(file_path)
    
    def display_image(self, image_path):
        """Display an image in the UI"""
        # Open and resize the image for display
        pil_image = Image.open(image_path)
        pil_image = pil_image.resize((400, 600), Image.LANCZOS)
        
        # Convert to Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update the image label
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image  # Keep a reference
    
    def analyze_single_image(self):
        """Analyze the currently selected image"""
        if not hasattr(self, 'current_image_path'):
            messagebox.showwarning("Warning", "Please select an image first.")
            return
            
        self.status_var.set("Analyzing image...")
        self.root.update()
        
        try:
            # Process the image
            result = self.analyzer.process_image(self.current_image_path)
            
            if not result['success']:
                messagebox.showerror("Analysis Error", result['error'])
                self.status_var.set("Analysis failed.")
                return
                
            # Display the image with bounding box
            bbox_image = self.visualizer.draw_bounding_box(result['image'], result['bbox'])
            
            # Convert OpenCV image to PIL and then to Tkinter PhotoImage
            bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(bbox_image_rgb)
            pil_image = pil_image.resize((400, 600), Image.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Update the image label
            self.image_label.configure(image=tk_image)
            self.image_label.image = tk_image  # Keep a reference
            
            # Display features
            features = result['features']
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Extracted Features:\n\n")
            
            # Format and display key features
            feature_text = f"Left Margin: {features['left_margin']:.1f} pixels ({features['left_margin_good']})\n"
            feature_text += f"Right Margin: {features['right_margin']:.1f} pixels ({features['right_margin_good']})\n"
            feature_text += f"Top Margin: {features['top_margin']:.1f} pixels ({features['top_margin_good']})\n"
            feature_text += f"Bottom Margin: {features['bottom_margin']:.1f} pixels ({features['bottom_margin_good']})\n\n"
            
            feature_text += f"Line Characteristics:\n"
            feature_text += f"  - Straight: {features['is_line_straight']}\n"
            feature_text += f"  - Sloped: {features['is_line_sloped']}\n"
            feature_text += f"  - Curved: {features['is_line_curved']}\n"
            feature_text += f"  - Average Line Angle: {features['mean_line_angle']:.2f}°\n\n"
            
            feature_text += f"Word Spacing: {features['avg_word_spacing']:.2f} pixels\n"
            feature_text += f"Text Density: {features['text_density']:.4f}\n"
            feature_text += f"Line Height Variation: {features['line_height_variation']:.4f}\n"
            
            self.results_text.insert(tk.END, feature_text)
            
            # Predict personality traits
            if self.predictor.model:
                ml_prediction = self.predictor.predict_ml_traits(features)
                if ml_prediction:
                    traits = ml_prediction
                else:
                    traits = self.predictor.predict_basic_traits(features)
            else:
                traits = self.predictor.predict_basic_traits(features)
                
            # Display personality analysis
            self.personality_text.delete(1.0, tk.END)
            self.personality_text.insert(tk.END, "Personality Analysis:\n\n")
            
            personality_desc = self.predictor.get_personality_description(traits)
            self.personality_text.insert(tk.END, personality_desc)
            
            # Save the analysis result
            self.data_manager.save_analysis_result(
                self.current_image_path, features, traits)
                
            self.status_var.set("Analysis complete.")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Analysis failed.")
    
    def select_batch_folder(self):
        """Open folder dialog to select a directory with images"""
        folder_path = filedialog.askdirectory(title="Select Folder with Handwriting Samples")
        
        if folder_path:
            self.batch_folder_path = folder_path
            self.batch_status_var.set(f"Selected folder: {os.path.basename(folder_path)}")
            
            # Count image files
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(image_extensions)]
            
            self.batch_status_var.set(f"Found {len(image_files)} images in folder.")
    
    def process_batch(self):
        """Process all images in the selected folder"""
        if not hasattr(self, 'batch_folder_path'):
            messagebox.showwarning("Warning", "Please select a folder first.")
            return
            
        # Get image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_files = [f for f in os.listdir(self.batch_folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            messagebox.showinfo("Info", "No image files found in the selected folder.")
            return
            
        # Clear existing results
        for i in self.batch_results_tree.get_children():
            self.batch_results_tree.delete(i)
            
        # Reset progress bar
        self.batch_progress['value'] = 0
        self.batch_progress['maximum'] = len(image_files)
        
        # Process each image
        for i, filename in enumerate(image_files):
            file_path = os.path.join(self.batch_folder_path, filename)
            self.batch_status_var.set(f"Processing {i+1}/{len(image_files)}: {filename}")
            self.batch_progress['value'] = i + 1
            self.root.update()
            
            try:
                # Process the image
                result = self.analyzer.process_image(file_path)
                
                if result['success']:
                    features = result['features']
                    
                    # Predict personality traits
                    if self.predictor.model:
                        ml_prediction = self.predictor.predict_ml_traits(features)
                        if ml_prediction:
                            traits = ml_prediction
                        else:
                            traits = self.predictor.predict_basic_traits(features)
                    else:
                        traits = self.predictor.predict_basic_traits(features)
                        
                    # Save the result
                    self.data_manager.save_analysis_result(file_path, features, traits)
                    
                    # Add to treeview
                    self.batch_results_tree.insert('', tk.END, values=(
                        filename,
                        traits.get('personality_type', 'Unknown'),
                        traits.get('family_attachment', False),
                        traits.get('risk_taking', False),
                        traits.get('energy', False),
                        traits.get('planning', False)
                    ))
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        self.batch_status_var.set(f"Batch processing complete. Processed {len(image_files)} images.")
    
    def export_batch_results(self):
        """Export batch results to CSV"""
        if not self.batch_results_tree.get_children():
            messagebox.showinfo("Info", "No results to export.")
            return
            
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Create DataFrame from treeview data
            data = []
            columns = ["filename", "personality_type", "family_attachment", 
                      "risk_taking", "energy", "planning"]
                      
            for item_id in self.batch_results_tree.get_children():
                values = self.batch_results_tree.item(item_id)['values']
                data.append(dict(zip(columns, values)))
                
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def refresh_data(self):
        """Refresh the data display"""
        # Clear existing data tree
        for i in self.data_tree.get_children():
            self.data_tree.delete(i)
            
        # Get current data
        features_df = self.data_manager.get_all_features()
        
        if len(features_df) == 0:
            self.data_stats_text.delete(1.0, tk.END)
            self.data_stats_text.insert(tk.END, "No data available.")
            return
            
        # Configure columns
        columns = list(features_df.columns)
        self.data_tree['columns'] = columns
        
        # Hide default first column
        self.data_tree['show'] = 'headings'
        
        # Set up column headings
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
            
        # Add data to treeview
        for i, row in features_df.iterrows():
            values = [row[col] for col in columns]
            self.data_tree.insert('', tk.END, values=values)
            
        # Update statistics
        self.data_stats_text.delete(1.0, tk.END)
        stats_text = f"Total samples: {len(features_df)}\n"
        
        # Count personality types
        if 'trait_personality_type' in features_df.columns:
            personality_counts = features_df['trait_personality_type'].value_counts()
            stats_text += "Personality types:\n"
            for personality, count in personality_counts.items():
                stats_text += f"  - {personality}: {count}\n"
                
        self.data_stats_text.insert(tk.END, stats_text)
        
        # Update visualization options
        self.update_visualization_options()
    
    def update_visualization_options(self):
        """Update the available options for visualization"""
        features_df = self.data_manager.get_all_features()
        
        # Get available numerical features
        numerical_features = features_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Update feature dropdown
        self.feature_dropdown['values'] = numerical_features
        if numerical_features:
            self.feature_dropdown.current(0)
    
    def update_visualization(self, event=None):
        """Update UI based on selected visualization type"""
        viz_type = self.viz_type_var.get()
        
        if viz_type == "Feature Distributions":
            self.feature_dropdown.pack(side=tk.LEFT, padx=5)
            features_df = self.data_manager.get_all_features()
            numerical_features = features_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            self.feature_dropdown['values'] = numerical_features
            
        elif viz_type == "Trait Distributions":
            self.feature_dropdown.pack(side=tk.LEFT, padx=5)
            features_df = self.data_manager.get_all_features()
            trait_columns = [col for col in features_df.columns if col.startswith('trait_')]
            self.feature_dropdown['values'] = [col.replace('trait_', '') for col in trait_columns]
            
        else:
            self.feature_dropdown.pack_forget()
    
    def generate_visualization(self):
        """Generate the selected visualization"""
        viz_type = self.viz_type_var.get()
        features_df = self.data_manager.get_all_features()
        
        if len(features_df) == 0:
            messagebox.showinfo("Info", "No data available for visualization.")
            return
            
        # Clear previous visualization
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
            
        try:
            if viz_type == "Feature Distributions":
                feature = self.feature_var.get()
                if not feature:
                    messagebox.showwarning("Warning", "Please select a feature.")
                    return
                    
                fig = self.visualizer.create_feature_histogram(features_df, feature)
                
                # Display the figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            elif viz_type == "Trait Distributions":
                trait = self.feature_var.get()
                if not trait:
                    messagebox.showwarning("Warning", "Please select a trait.")
                    return
                    
                fig = self.visualizer.create_trait_distribution_pie(features_df, trait)
                
                if fig:
                    # Display the figure
                    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                    canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                else:
                    messagebox.showinfo("Info", f"No data available for trait '{trait}'.")
                    
            elif viz_type == "Feature Correlations":
                fig = self.visualizer.create_correlation_heatmap(features_df)
                
                # Display the figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            elif viz_type == "Personality Type Distribution":
                if 'trait_personality_type' in features_df.columns:
                    fig = self.visualizer.create_trait_distribution_pie(features_df, 'personality_type')
                    
                    if fig:
                        # Display the figure
                        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                        canvas = FigureCanvasTkAgg(fig, master=self.viz_canvas_frame)
                        canvas.draw()
                        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    else:
                        messagebox.showinfo("Info", "No personality type data available.")
                else:
                    messagebox.showinfo("Info", "No personality type data available.")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")
    
    def train_new_model(self):
        """Train a new ML model on the collected data"""
        features_df = self.data_manager.get_all_features()
        
        if len(features_df) < 10:
            messagebox.showwarning("Warning", "Not enough data to train a model. Need at least 10 samples.")
            return
            
        # Check if personality type data is available
        if 'trait_personality_type' not in features_df.columns:
            messagebox.showwarning("Warning", "No personality type data available for training.")
            return
            
        try:
            # Select feature columns and target
            feature_cols = [col for col in features_df.columns if col not in 
                          ['image_path', 'timestamp'] and not col.startswith('trait_')]
                          
            X = features_df[feature_cols]
            y = features_df['trait_personality_type']
            
            # Train the model
            training_result = self.predictor.train_model(X, y)
            
            # Show results
            accuracy = training_result['accuracy']
            report = training_result['report']
            
            # Save the model
            if not os.path.exists('models'):
                os.makedirs('models')
                
            model_path = os.path.join('models', 'personality_model.pkl')
            self.predictor.save_model(model_path)
            
            # Show summary
            summary = f"Model trained successfully!\n\n"
            summary += f"Accuracy: {accuracy:.4f}\n\n"
            summary += f"Classification Report:\n{report}\n\n"
            summary += f"Model saved to: {model_path}"
            
            # Create a dialog to show results
            result_dialog = tk.Toplevel(self.root)
            result_dialog.title("Model Training Results")
            result_dialog.geometry("500x400")
            
            result_text = tk.Text(result_dialog, wrap=tk.WORD)
            result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            result_text.insert(tk.END, summary)
            
            close_btn = ttk.Button(result_dialog, text="Close", command=result_dialog.destroy)
            close_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")


def main():
    """Main function to run the application"""
    # Create folders if they don't exist
    for folder in ['data', 'models', 'reports']:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    # Create and run the application
    root = tk.Tk()
    app = HandwritingAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()