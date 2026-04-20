# TransReID Player Matcher

## Project Title
**TransReID Player Matcher**

## Description
TransReID Player Matcher provides a robust solution for player image matching using deep learning techniques. It is designed to identify players in various scenarios by comparing uploaded images against a database of training images.

## Quick Start
### Installation Steps
1. **Clone the repository**:
   ```
   git clone https://github.com/51A96/TransReID-Player-Matcher.git
   cd TransReID-Player-Matcher
   ```
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

### Configuration Instructions
- Modify the configuration file as per your environment settings, specifying paths to the training datasets and model parameters.

### Feature Library Building
- Run the feature extraction script to build your feature library:
   ```
   python extract_features.py
   ```

### App Startup
- Start the application:
   ```
   python app.py
   ```

## Usage Flow
1. **Upload Image**: Select and upload an image of the player you wish to match.
2. **Set Parameters**: Configure any parameters that might affect the matching (e.g., threshold settings).
3. **Click Match**: Initiate the matching process by clicking the match button.
4. **View Results**: Review the results displayed after processing.

### Output Information
- **Player ID Extraction**: The unique identifier for each matched player.
- **Training Set Image ID**: Identifier for the training image used for comparison.
- **Similarity Scores**: Scores indicating how closely the uploaded image matches the training images (range: 0 to 1, where 1 is a perfect match).
- **Feature Statistics**: Detailed statistics of the features extracted from both the uploaded image and training matches.

## File Structure
```
TransReID-Player-Matcher/
├── app.py
├── extract_features.py
├── requirements.txt
├── config.yaml
└── README.md
```

## Troubleshooting Tips
- Ensure that all dependencies are correctly installed as per the requirements.txt file.
- If no matches are found, check if the uploaded image is clear and of good quality.
- Adjust similarity thresholds if the app is giving too many false positives or negatives.

For more information, please refer to the documentation provided in the repository.