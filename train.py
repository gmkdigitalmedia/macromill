"""
Sentiment Classification Training Pipeline
Enterprise-grade implementation for IMDb movie review sentiment analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from tqdm.auto import tqdm

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "archive/IMDB Dataset.csv"
MODEL_NAME = "distilbert-base-uncased"  # Efficient transformer model
OUTPUT_DIR = "models"
RESULTS_DIR = "results"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDb reviews"""

    def __init__(self, encodings: Dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class SentimentAnalyzer:
    """Main class for sentiment analysis pipeline"""

    def __init__(self, data_path: str, model_name: str):
        self.data_path = data_path
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Create output directories
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        Path(RESULTS_DIR).mkdir(exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.df = None

    def load_and_explore_data(self) -> pd.DataFrame:
        """Load dataset and perform exploratory data analysis"""
        logger.info("Loading dataset...")
        self.df = pd.read_csv(self.data_path)

        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"\nDataset info:\n{self.df.info()}")
        logger.info(f"\nFirst few rows:\n{self.df.head()}")
        logger.info(f"\nSentiment distribution:\n{self.df['sentiment'].value_counts()}")

        # Check for missing values
        logger.info(f"\nMissing values:\n{self.df.isnull().sum()}")

        # Basic statistics
        self.df['review_length'] = self.df['review'].apply(len)
        self.df['word_count'] = self.df['review'].apply(lambda x: len(x.split()))

        logger.info(f"\nReview length statistics:\n{self.df['review_length'].describe()}")
        logger.info(f"\nWord count statistics:\n{self.df['word_count'].describe()}")

        return self.df

    def visualize_data(self):
        """Create visualizations for EDA"""
        logger.info("Creating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Sentiment distribution
        sentiment_counts = self.df['sentiment'].value_counts()
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, color=['#ff6b6b', '#4ecdc4'])
        axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Sentiment')
        axes[0, 0].set_ylabel('Count')
        for i, v in enumerate(sentiment_counts.values):
            axes[0, 0].text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')

        # 2. Review length distribution by sentiment
        self.df.boxplot(column='review_length', by='sentiment', ax=axes[0, 1])
        axes[0, 1].set_title('Review Length Distribution by Sentiment', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Sentiment')
        axes[0, 1].set_ylabel('Review Length (characters)')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=0)

        # 3. Word count distribution by sentiment
        self.df.boxplot(column='word_count', by='sentiment', ax=axes[1, 0])
        axes[1, 0].set_title('Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Sentiment')
        axes[1, 0].set_ylabel('Word Count')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=0)

        # 4. Review length histogram
        axes[1, 1].hist([
            self.df[self.df['sentiment'] == 'positive']['word_count'],
            self.df[self.df['sentiment'] == 'negative']['word_count']
        ], bins=50, label=['Positive', 'Negative'], alpha=0.7, color=['#4ecdc4', '#ff6b6b'])
        axes[1, 1].set_title('Word Count Histogram by Sentiment', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 1000)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/eda_visualizations.png', dpi=300, bbox_inches='tight')
        logger.info(f"Visualizations saved to {RESULTS_DIR}/eda_visualizations.png")
        plt.close()

    def prepare_data(self) -> Tuple:
        """Prepare and tokenize data for training"""
        logger.info("Preparing data for training...")

        # Convert sentiment to binary labels
        self.df['label'] = self.df['sentiment'].map({'positive': 1, 'negative': 0})

        # Split data: 80% train, 10% validation, 10% test
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            self.df['review'].tolist(),
            self.df['label'].tolist(),
            test_size=0.2,
            random_state=SEED,
            stratify=self.df['label']
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=SEED,
            stratify=temp_labels
        )

        logger.info(f"Train size: {len(train_texts)}")
        logger.info(f"Validation size: {len(val_texts)}")
        logger.info(f"Test size: {len(test_texts)}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        test_encodings = self.tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )

        # Create datasets
        train_dataset = IMDBDataset(train_encodings, train_labels)
        val_dataset = IMDBDataset(val_encodings, val_labels)
        test_dataset = IMDBDataset(test_encodings, test_labels)

        return train_dataset, val_dataset, test_dataset, test_texts, test_labels

    def compute_metrics(self, pred) -> Dict:
        """Compute evaluation metrics"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train_model(self, train_dataset: Dataset, val_dataset: Dataset):
        """Train the sentiment classification model"""
        logger.info("Initializing model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.to(self.device)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{RESULTS_DIR}/logs',
            logging_steps=100,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            learning_rate=LEARNING_RATE,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            report_to='none',  # Disable wandb/tensorboard
            seed=SEED
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # Train model
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save model and tokenizer
        logger.info("Saving model...")
        self.model.save_pretrained(f'{OUTPUT_DIR}/final_model')
        self.tokenizer.save_pretrained(f'{OUTPUT_DIR}/final_model')

        # Log training results
        logger.info(f"Training completed. Results: {train_result.metrics}")

        return trainer, train_result

    def evaluate_model(self, trainer: Trainer, test_dataset: Dataset,
                       test_texts: List[str], test_labels: List[int]):
        """Comprehensive model evaluation"""
        logger.info("Evaluating model on test set...")

        # Get predictions
        predictions = trainer.predict(test_dataset)
        preds = predictions.predictions.argmax(-1)
        probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()

        # Calculate metrics
        accuracy = accuracy_score(test_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average='binary'
        )
        roc_auc = roc_auc_score(test_labels, probs)

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }

        logger.info(f"\nTest Set Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")

        # Classification report
        logger.info(f"\nClassification Report:\n{classification_report(test_labels, preds, target_names=['Negative', 'Positive'])}")

        # Save results
        with open(f'{RESULTS_DIR}/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Create visualizations
        self._create_evaluation_plots(test_labels, preds, probs)

        return results

    def _create_evaluation_plots(self, y_true: List[int], y_pred: List[int], y_probs: np.ndarray):
        """Create evaluation visualizations"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/evaluation_metrics.png', dpi=300, bbox_inches='tight')
        logger.info(f"Evaluation plots saved to {RESULTS_DIR}/evaluation_metrics.png")
        plt.close()

    def test_sample_predictions(self):
        """Test model with sample predictions"""
        logger.info("\nTesting sample predictions...")

        samples = [
            "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
            "Terrible waste of time. Poor acting, bad script, and boring storyline.",
            "It was okay, nothing special but not terrible either.",
            "One of the best films I've ever seen! Highly recommend!",
            "Awful movie. I want my money back."
        ]

        self.model.eval()
        results = []

        for text in samples:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][prediction].item()

            sentiment = 'positive' if prediction == 1 else 'negative'
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })

            logger.info(f"\nText: {text}")
            logger.info(f"Prediction: {sentiment} (confidence: {confidence:.4f})")

        # Save sample predictions
        with open(f'{RESULTS_DIR}/sample_predictions.json', 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("SENTIMENT CLASSIFICATION PIPELINE - IMDb Movie Reviews")
    logger.info("="*80)

    # Initialize analyzer
    analyzer = SentimentAnalyzer(DATA_PATH, MODEL_NAME)

    # 1. Load and explore data
    logger.info("\n[STEP 1/5] Data Loading and Exploration")
    analyzer.load_and_explore_data()

    # 2. Create visualizations
    logger.info("\n[STEP 2/5] Creating Visualizations")
    analyzer.visualize_data()

    # 3. Prepare data
    logger.info("\n[STEP 3/5] Data Preparation")
    train_dataset, val_dataset, test_dataset, test_texts, test_labels = analyzer.prepare_data()

    # 4. Train model
    logger.info("\n[STEP 4/5] Model Training")
    trainer, train_result = analyzer.train_model(train_dataset, val_dataset)

    # 5. Evaluate model
    logger.info("\n[STEP 5/5] Model Evaluation")
    results = analyzer.evaluate_model(trainer, test_dataset, test_texts, test_labels)

    # Test sample predictions
    analyzer.test_sample_predictions()

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Model saved to: {OUTPUT_DIR}/final_model")
    logger.info(f"Results saved to: {RESULTS_DIR}/")
    logger.info("="*80)


if __name__ == "__main__":
    main()
