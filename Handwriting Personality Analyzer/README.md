# Handwriting Personality Analyzer

This application analyzes handwritten text samples to predict personality traits based on spatial features like margins, line orientation, word spacing, and other measurable characteristics. It combines computer vision techniques with machine learning to provide personality insights.

## Features

- **Image Analysis**: Extract spatial features from handwriting samples
- **Personality Prediction**: Predict personality traits using both rule-based and ML-based approaches
- **User-Friendly Interface**: Easy-to-use GUI for single image analysis and batch processing
- **Data Management**: Store and manage analyzed handwriting samples
- **Visualization**: Generate charts and graphs to visualize the data
- **Model Training**: Train machine learning models on collected data

## Requirements

- Python 3.7+
- Required packages:
  - opencv-python (for image processing)
  - numpy (for numerical operations)
  - pandas (for data management)
  - matplotlib (for visualization)
  - seaborn (for enhanced visualization)
  - scikit-learn (for machine learning)
  - pillow (for image handling)

Install dependencies using:
```
pip install opencv-python numpy pandas matplotlib seaborn scikit-learn pillow
```

## Project Structure

```
handwriting-personality-analyzer/
│
├── data/                 # Data storage directory
│   └── features.csv      # Extracted features from samples
│   └── analysis_results.json  # Detailed analysis results
│
├── models/               # Trained ML models
│   └── personality_model.pkl  # Saved personality prediction model
│
├── reports/              # Generated reports
│
└── main.py               # Main application file
```

## Usage

1. **Running the Application**:
   ```
   python main.py
   ```

2. **Single Image Analysis**:
   - Go to the "Analysis" tab
   - Click "Browse..." to select a handwriting sample
   - Click "Analyze" to process the image
   - View extracted features and personality predictions

3. **Batch Processing**:
   - Go to the "Batch Processing" tab
   - Select a folder containing multiple handwriting samples
   - Click "Process Batch" to analyze all images
   - Export results to CSV if needed

4. **Data Management**:
   - The "Data Management" tab shows all collected data
   - Use "Refresh Data" to update the view
   - Train models on the collected data

5. **Statistics and Visualization**:
   - View different visualizations in the "Statistics" tab
   - Analyze distributions of features and personality traits
   - Examine correlations between different features

## Personality Trait Interpretation

The system analyzes the following personality traits:

1. **Family Attachment**:
   - Strong left margin: Independence, weaker family bonds
   - Small/no left margin: Strong attachment to family and traditions

2. **Risk Taking**:
   - Large right margin: Cautiousness, fear of failure
   - Small/no right margin: Risk-taking, adventurous nature

3. **Energy Level**:
   - Sloped writing: Positive energy, optimism
   - Straight writing: More reserved energy

4. **Planning**:
   - Large bottom margin: Productivity, good resource utilization
   - Small/no bottom margin: Impulsive but detailed planner

5. **Personality Type**:
   - Introvert, Extrovert, or Ambivert classification based on combined features

## Extending the System

### Adding New Features

To add new handwriting features for analysis:

1. Create a new extraction method in the `HandwritingAnalyzer` class
2. Update the `process_image` method to include your new features
3. Modify the personality prediction logic in `PersonalityPredictor` class

### Improving Machine Learning

1. Collect more labeled samples
2. Try different ML algorithms (Random Forest, SVM, Neural Networks)
3. Fine-tune hyperparameters for better accuracy

## Data Science and Big Data Analytics Integration

This project exemplifies DSBDA principles through:

1. **Feature Engineering**: Extraction of meaningful features from raw handwriting images
2. **Data Collection and Management**: Systematic storage and organization of analyzed samples
3. **Predictive Modeling**: Using ML algorithms to predict personality traits
4. **Data Visualization**: Visual representation of features and predictions
5. **Statistical Analysis**: Analyzing distributions and correlations in the data

## Citation

If using this software for research or publications, please cite:

```
Handwriting Personality Analyzer
A Data Science and Big Data Analytics project combining computer vision and machine learning
for handwriting-based personality prediction.
```

## License

This project is available under the MIT License.

## 👤 Author

- Nishant Kadlak
- Vivek Janbandhu
- Prathamesh Khokaralkar